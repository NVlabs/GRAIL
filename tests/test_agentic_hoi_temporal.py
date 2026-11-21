"""Behavioral checks for motion priors on HOI corrections, not post-hoc filtering."""

import pytest
import torch

from grail.agentic_kinematic.hoi_temporal import correction_motion_loss

UNIT_SCALES = dict(
    velocity_scale_m_s=1.0,
    acceleration_scale_m_s2=1.0,
    jerk_scale_m_s3=1.0,
)


def x_motion(values, *, requires_grad=False):
    result = torch.zeros((len(values), 1, 3), dtype=torch.float64)
    result[:, 0, 0] = torch.tensor(values, dtype=torch.float64)
    return result.requires_grad_(requires_grad)


def test_constant_correction_preserves_a_moving_reference():
    reference = x_motion([0, 1, 4, 9, 16], requires_grad=True)
    current = (reference.detach() + torch.tensor([0.5, -2.0, 4.0])).requires_grad_()
    loss = correction_motion_loss(current, reference, fps=24)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(current.grad) == 0
    assert reference.grad is None


def test_unchanged_jittery_reference_is_not_filtered():
    reference = x_motion([0, 5, -2, 8, 1])
    current = reference.clone().requires_grad_()
    loss = correction_motion_loss(current, reference, fps=24)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(current.grad) == 0


def test_abrupt_correction_couples_both_sides_of_transition():
    reference = x_motion([0, 0, 0, 0], requires_grad=True)
    current = x_motion([0, 0, 1, 1], requires_grad=True)
    loss = correction_motion_loss(current, reference, fps=1, peak_weight=0, **UNIT_SCALES)
    # Velocity: one unit step over three intervals. Acceleration: two
    # opposite unit changes. Jerk: one change from +1 to -1.
    assert loss.item() == pytest.approx(1 / 3 + 1 + 4)
    loss.backward()
    assert current.grad[1, 0, 0] < 0  # Gradient descent lifts the pre-step value.
    assert current.grad[2, 0, 0] > 0  # And lowers the post-step value.
    assert torch.isfinite(current.grad).all()
    assert current.grad[..., 1:].count_nonzero() == 0
    assert reference.grad is None


def test_fps_scales_velocity_in_physical_time():
    current = x_motion([0, 1, 2, 3])
    reference = torch.zeros_like(current)
    low = correction_motion_loss(current, reference, fps=2, peak_weight=0.1, **UNIT_SCALES)
    high = correction_motion_loss(current, reference, fps=4, peak_weight=0.1, **UNIT_SCALES)
    assert low.item() == pytest.approx(4.4)
    assert high.item() == pytest.approx(4 * low.item())


@pytest.mark.parametrize("frames", [0, 1])
def test_empty_or_single_frame_has_graph_connected_zero(frames):
    current = torch.zeros((frames, 2, 3), dtype=torch.float64, requires_grad=True)
    reference = current.detach().clone().requires_grad_()
    loss = correction_motion_loss(current, reference, fps=24)
    assert loss.item() == 0
    loss.backward()
    assert current.grad is not None and current.grad.shape == current.shape
    assert current.grad.count_nonzero() == 0
    assert reference.grad is None


@pytest.mark.parametrize("frames", [2, 3])
def test_short_sequence_uses_only_available_derivatives(frames):
    current = x_motion(list(range(frames)), requires_grad=True)
    loss = correction_motion_loss(
        current, torch.zeros_like(current), fps=1, peak_weight=0, **UNIT_SCALES
    )
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert torch.isfinite(current.grad).all()


def test_peak_term_keeps_an_isolated_spike_visible_in_long_sequences():
    results = []
    for frames in (8, 80):
        current = x_motion([0, 0, 1, *([0] * (frames - 3))])
        reference = torch.zeros_like(current)
        mean_only = correction_motion_loss(current, reference, fps=1, peak_weight=0, **UNIT_SCALES)
        with_peak = correction_motion_loss(
            current, reference, fps=1, peak_weight=0.25, **UNIT_SCALES
        )
        results.append((mean_only.item(), (with_peak - mean_only).item()))
    assert results[1][0] < results[0][0]
    assert results[1][1] == pytest.approx(results[0][1])
    assert results[1][1] > 0


@pytest.mark.parametrize(
    "key",
    ["fps", "velocity_scale_m_s", "acceleration_scale_m_s2", "jerk_scale_m_s3"],
)
@pytest.mark.parametrize("invalid", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_time_and_physical_scales_must_be_finite_positive_numbers(key, invalid):
    options = dict(fps=24, **UNIT_SCALES)
    options[key] = invalid
    current = x_motion([0, 0])
    with pytest.raises(ValueError):
        correction_motion_loss(current, current, **options)


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "0.1"])
def test_peak_weight_must_be_finite_nonnegative_number(invalid):
    current = x_motion([0, 0])
    with pytest.raises(ValueError):
        correction_motion_loss(current, current, fps=24, peak_weight=invalid)


@pytest.mark.parametrize(
    "current,reference",
    [
        (torch.zeros(3, 3), torch.zeros(3, 3)),
        (torch.zeros(3, 1, 4), torch.zeros(3, 1, 4)),
        (torch.zeros(3, 1, 3), torch.zeros(4, 1, 3)),
        (torch.zeros(3, 1, 3), torch.zeros(3, 2, 3)),
        (torch.zeros(3, 1, 3, dtype=torch.int64), torch.zeros(3, 1, 3)),
        (torch.zeros(3, 1, 3), torch.zeros(3, 1, 3, dtype=torch.int64)),
        (torch.zeros(3, 1, 3), torch.zeros(3, 1, 3, dtype=torch.float64)),
    ],
)
def test_motion_tensor_contract_rejects_incompatible_inputs(current, reference):
    with pytest.raises(ValueError):
        correction_motion_loss(current, reference, fps=24)
