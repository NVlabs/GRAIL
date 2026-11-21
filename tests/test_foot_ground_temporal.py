"""Fixed foot-label transitions smooth corrections without inventing support."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from grail.optimization.foot_ground import FOOT_PARTS
from grail.optimization.foot_ground_temporal import FootGroundTemporal
from grail.optimization.loss_computer import LossComputer

REGIONS = tuple(torch.arange(i * 3, i * 3 + 3) for i in range(4))


def geometry(frames=20):
    return torch.zeros(frames, 12, 3, dtype=torch.double)


def labels(frames=20):
    probabilities = torch.zeros(frames, 4, dtype=torch.double)
    probabilities[6:12, 1] = 0.9  # Left toe only, with touchdown and liftoff.
    return probabilities


def helper(reference=None, probabilities=None, **options):
    reference = geometry() if reference is None else reference
    probabilities = labels(len(reference)) if probabilities is None else probabilities
    return FootGroundTemporal(
        reference, REGIONS, probabilities, fps=options.pop("fps", 24), **options
    )


def test_events_follow_strict_independent_labels_without_inventing_endpoint_events():
    probabilities = labels()
    probabilities[:, 0] = 1  # Continuous support has no touchdown inside the clip.
    probabilities[:, 2] = 0.5  # Threshold equality is unsupported.
    probabilities[2:5, 3] = 0.50001
    constraint = helper(probabilities=probabilities)
    assert constraint.events == ((), (6, 12), (), (2, 5))
    assert constraint.acceleration_weights[5, 1] == 1  # Source frame 6.
    assert constraint.acceleration_weights[11, 1] == 1  # First unsupported frame12.
    assert constraint.acceleration_weights[:, 0].sum() == 0
    assert constraint.acceleration_weights[:, 2].sum() == 0


def test_bounded_windows_taper_and_overlaps_do_not_double_count():
    probabilities = torch.zeros(20, 4, dtype=torch.double)
    probabilities[7:9, 1] = 1
    constraint = helper(probabilities=probabilities, window_frames=3)
    weights = constraint.acceleration_weights[:, 1]
    assert weights[6] == weights[8] == 1
    assert 0 < weights[4] < weights[5] < weights[6]
    assert weights[:3].sum() == weights[11:].sum() == 0
    assert constraint.velocity_weights.max() <= 1
    assert weights.max() <= 1


def test_existing_constant_velocity_swing_and_height_offset_are_free():
    reference = geometry()
    reference[:, :, 0] = torch.arange(20, dtype=torch.double)[:, None] * 0.06
    reference[:, :, 2] = 1.8  # Temporal loss must not introduce floor attraction.
    constraint = helper(reference)
    values = constraint.terms(reference + torch.tensor([1.0, -0.2, 0.4]))
    assert values["window_velocity"].item() == 0
    assert values["window_acceleration"].item() == 0


def test_preexisting_touchdown_and_liftoff_spikes_receive_actual_acceleration_penalty():
    for event in (6, 12):
        reference = geometry()
        reference[event:, 3:6, 2] = 0.1
        values = helper(reference).terms(reference)
        assert values["window_velocity"].item() == 0
        assert values["window_acceleration"].item() > 0


def test_correction_speed_is_relative_to_frozen_starting_motion():
    reference = geometry()
    reference[:, 3:6, 0] = torch.arange(20, dtype=torch.double)[:, None] * 0.1
    constraint = helper(reference)
    candidate = reference.clone()
    candidate[6:, 3:6, 0] += 0.05
    assert constraint.terms(candidate)["window_velocity"] > 0
    # Changing the input tensor after initialization cannot rewrite the reference.
    reference.copy_(candidate)
    assert constraint.terms(candidate)["window_velocity"] > 0


def test_velocity_and_acceleration_have_physical_source_fps_units():
    candidate = geometry()
    candidate[:, 3:6, 0] = torch.arange(20, dtype=torch.double)[:, None] * 0.01
    options = dict(correction_speed_tolerance_mps=0, velocity_scale_mps=1)
    slow = helper(fps=10, **options).terms(candidate)
    fast = helper(fps=20, **options).terms(candidate)
    assert slow["window_velocity"].item() == pytest.approx(0.5 * 0.1**2)
    assert fast["window_velocity"].item() == pytest.approx(4 * slow["window_velocity"].item())
    candidate[:, 3:6, 0] = torch.arange(20, dtype=torch.double)[:, None].square() * 0.0001
    options = dict(acceleration_tolerance_mps2=0, acceleration_scale_mps2=1)
    slow = helper(fps=10, **options).terms(candidate)
    fast = helper(fps=20, **options).terms(candidate)
    assert slow["window_acceleration"].item() == pytest.approx(0.5 * 0.02**2)
    assert fast["window_acceleration"].item() == pytest.approx(
        16 * slow["window_acceleration"].item()
    )


def test_only_transition_region_and_local_stencils_receive_gradient():
    torch.manual_seed(3)
    candidate = (geometry() + torch.randn_like(geometry()) * 0.1).requires_grad_()
    values = helper(window_frames=2).terms(candidate)
    sum(values.values()).backward()
    assert candidate.grad[4:15, 3:6].abs().sum() > 0
    assert candidate.grad[:4].abs().sum() == candidate.grad[15:].abs().sum() == 0
    assert candidate.grad[:, :3].abs().sum() == candidate.grad[:, 6:].abs().sum() == 0
    assert torch.isfinite(candidate.grad).all()


def test_labels_and_region_indices_are_frozen_and_not_differentiable():
    probabilities = labels().requires_grad_()
    regions = tuple(ids.clone() for ids in REGIONS)
    constraint = FootGroundTemporal(geometry(), regions, probabilities, fps=24)
    with torch.no_grad():
        probabilities.zero_()
        regions[1].fill_(0)
    candidate = geometry()
    candidate[6, 3:6, 0] = 0.1
    candidate.requires_grad_()
    sum(constraint.terms(candidate).values()).backward()
    assert candidate.grad[:, 3:6].abs().sum() > 0
    assert probabilities.grad is None


@pytest.mark.parametrize("frames", [1, 2, 3, 8])
@pytest.mark.parametrize("contact", [0.0, 1.0])
def test_no_transitions_including_short_sequences_are_differentiable_zero(frames, contact):
    candidate = torch.randn_like(geometry(frames)).requires_grad_()
    probabilities = torch.full((frames, 4), contact)
    constraint = helper(geometry(frames), probabilities)
    values = constraint.terms(candidate)
    assert all(value.item() == 0 for value in values.values())
    sum(values.values()).backward()
    assert candidate.grad.abs().sum() == 0


def test_two_frame_transition_has_velocity_but_no_acceleration_stencil():
    probabilities = torch.zeros(2, 4)
    probabilities[1, 1] = 1
    constraint = helper(geometry(2), probabilities)
    candidate = geometry(2)
    candidate[1, 3:6, 2] = 0.1
    candidate.requires_grad_()
    values = constraint.terms(candidate)
    assert values["window_velocity"] > 0
    assert values["window_acceleration"] == 0
    sum(values.values()).backward()
    assert torch.isfinite(candidate.grad).all()


def test_gradient_matches_finite_differences():
    torch.manual_seed(31)
    candidate = (geometry(8) + torch.randn_like(geometry(8)) * 0.07).requires_grad_()
    probabilities = torch.zeros(8, 4)
    probabilities[3:6, 1] = 1
    constraint = helper(geometry(8), probabilities)
    assert torch.autograd.gradcheck(
        lambda value: sum(constraint.terms(value).values()),
        (candidate,),
        eps=1e-6,
        atol=2e-5,
        fast_mode=True,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"fps": "source"},
        {"fps": 0},
        {"fps": True},
        {"fps": float("inf")},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"window_frames": 0},
        {"window_frames": 1.5},
        {"window_frames": True},
        {"velocity_scale_mps": 0},
        {"acceleration_scale_mps2": -1},
        {"correction_speed_tolerance_mps": -0.1},
        {"acceleration_tolerance_mps2": float("nan")},
    ],
)
def test_invalid_controls_fail(options):
    with pytest.raises(ValueError):
        helper(**options)


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        torch.zeros(19, 4),
        torch.zeros(20, 3),
        torch.zeros(20, 5),
        torch.zeros(20, 4, dtype=torch.complex64),
        torch.full((20, 4), float("nan")),
        torch.full((20, 4), -0.1),
        torch.full((20, 4), 1.1),
    ],
)
def test_missing_malformed_or_nonfinite_labels_fail(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        FootGroundTemporal(geometry(), REGIONS, probabilities, fps=24)


@pytest.mark.parametrize(
    "regions",
    [
        REGIONS[:3],
        (torch.tensor([]), *REGIONS[1:]),
        (torch.tensor([0.5]), *REGIONS[1:]),
        (torch.tensor([True]), *REGIONS[1:]),
        (torch.tensor([12]), *REGIONS[1:]),
        (torch.tensor([-1]), *REGIONS[1:]),
        (torch.tensor([0, 0]), *REGIONS[1:]),
        (REGIONS[1], *REGIONS[1:]),
    ],
)
def test_invalid_native_regions_fail(regions):
    with pytest.raises(ValueError):
        FootGroundTemporal(geometry(), regions, labels(), fps=24)


def test_native_layout_and_finites_fail_before_optimization():
    for vertices in (geometry(0), geometry().long(), geometry()[0], geometry() + float("nan")):
        with pytest.raises(ValueError, match="native vertices"):
            helper(vertices, labels(len(vertices)))
    constraint = helper()
    with pytest.raises(ValueError, match="layout"):
        constraint.terms(geometry(19))
    with pytest.raises(ValueError, match="layout"):
        constraint.terms(geometry().float())
    with pytest.raises(ValueError, match="finite"):
        constraint.terms(geometry() + float("inf"))


class NativeModel:
    def get_segment_indices(self, labels):
        part = FOOT_PARTS.index(labels[0])
        # Native SOMA heel segments contain their toe descendants.
        return torch.cat(REGIONS[part : part + 2]) if part in (0, 2) else REGIONS[part]

    def get_verts_segment(self, vertices, labels):
        return vertices[:, self.get_segment_indices(labels)]


def loss_config():
    return {
        "weight": 1.0,
        "component_weights": {"window_velocity": 0.1, "window_acceleration": 0.1},
        "options": {"fps": 24},
    }


def test_dispatch_retains_stance_and_ground_losses_and_caches_reference_per_dataset():
    computer = LossComputer(None, NativeModel(), "cpu", 2, None)
    data = SimpleNamespace(human=SimpleNamespace(foot_contact_probs=labels()))
    pred = SimpleNamespace(human=SimpleNamespace(verts_seq=geometry(), body_joints_seq=geometry()))
    losses = {
        "human_foot_ground_temporal": loss_config(),
        "human_foot_contact": {
            "weight": 30,
            "point_mode": "foot_parts",
            "velocity_space": "planar",
        },
        "human_foot_ground": {
            "weight": 300,
            "plane": {"normal": [0, 0, 1], "offset_m": 0, "source": "original floor"},
        },
    }
    initial, components = computer.compute_loss(data, pred, losses)
    assert initial == 0
    assert set(components) == set(losses)
    candidate = geometry()
    candidate[6:, 3:6, 0] = 0.05
    pred.human.verts_seq = candidate
    _, components = computer.compute_loss(data, pred, losses)
    assert components["human_foot_ground_temporal"] > 0
    assert len(computer._foot_ground_temporal_constraints) == 1
    cfg = deepcopy(losses["human_foot_ground_temporal"])
    cfg["component_weights"]["window_acceleration"] = 0
    old = computer._human_foot_ground_temporal_loss(data, pred, cfg, cfg["weight"])
    assert old > 0  # Reference was not replaced by candidate during the second evaluation.
    other_data = SimpleNamespace(human=SimpleNamespace(foot_contact_probs=labels()))
    new = computer._human_foot_ground_temporal_loss(other_data, pred, cfg, cfg["weight"])
    assert new == 0  # A different fit has its own initial geometry.
    assert len(computer._foot_ground_temporal_constraints) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": 1},
        {"component_weights": {}},
        {"component_weights": None},
        {"options": {}},
        {"options": {"fps": "source"}},
        {"options": {"fps": 24, "unknown": 1}},
        {"weight": float("nan")},
        {"weight": True},
        {"weight": -1},
        {"component_weights": {"window_velocity": 1, "window_acceleration": -1}},
    ],
)
def test_dispatch_rejects_ambiguous_or_invalid_configuration(change):
    computer = LossComputer(None, NativeModel(), "cpu", 2, None)
    data = SimpleNamespace(human=SimpleNamespace(foot_contact_probs=labels()))
    pred = SimpleNamespace(human=SimpleNamespace(verts_seq=geometry()))
    config = {**loss_config(), **change}
    with pytest.raises(ValueError):
        computer.compute_loss(data, pred, {"human_foot_ground_temporal": config})


def test_disabled_loss_does_not_require_labels_or_numeric_fps():
    computer = LossComputer(None, NativeModel(), "cpu", 2, None)
    candidate = geometry().requires_grad_()
    pred = SimpleNamespace(human=SimpleNamespace(verts_seq=candidate))
    config = {**loss_config(), "weight": 0, "options": {"fps": "source"}}
    total, _ = computer.compute_loss(None, pred, {"human_foot_ground_temporal": config})
    total.backward()
    assert candidate.grad.abs().sum() == 0


def test_predicted_and_reviewed_transition_schedules_cannot_both_be_enabled():
    computer = LossComputer(None, NativeModel(), "cpu", 2, None)
    with pytest.raises(ValueError, match="predicted foot-label and reviewed scene transitions"):
        computer.compute_loss(
            None,
            None,
            {
                "human_foot_ground_temporal": loss_config(),
                "human_scene_support_temporal": {"weight": 1},
            },
        )


def test_regular_optimizer_stage_reset_rebinds_seed_without_resetting_each_iteration():
    computer = LossComputer(None, NativeModel(), "cpu", 2, None)
    data = SimpleNamespace(human=SimpleNamespace(foot_contact_probs=labels()))
    pred = SimpleNamespace(human=SimpleNamespace(verts_seq=geometry()))
    cfg = loss_config()
    cfg["component_weights"]["window_acceleration"] = 0
    assert computer._human_foot_ground_temporal_loss(data, pred, cfg, 1) == 0
    pred.human.verts_seq = geometry()
    pred.human.verts_seq[6:, 3:6, 0] += 0.05
    first = computer._human_foot_ground_temporal_loss(data, pred, cfg, 1)
    assert first > 0
    assert computer._human_foot_ground_temporal_loss(data, pred, cfg, 1) == first
    old_reference = next(iter(computer._foot_ground_temporal_constraints.values()))
    computer.reset_foot_ground_temporal_reference()
    assert computer._foot_ground_temporal_constraints == {}
    assert computer._human_foot_ground_temporal_loss(data, pred, cfg, 1) == 0
    assert next(iter(computer._foot_ground_temporal_constraints.values())) is not old_reference
    pred.human.verts_seq[7:, 3:6, 0] += 0.05
    assert computer._human_foot_ground_temporal_loss(data, pred, cfg, 1) > 0
