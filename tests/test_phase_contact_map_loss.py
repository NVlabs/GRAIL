"""The regional solver consumes immutable unequal maps with live gradients."""

import copy
import json

import numpy as np
import pytest
import torch

from grail.optimization.fixed_contact import build_binding
from grail.optimization.phase_contact_map_loss import PhaseContactMapLoss, _partial_mass


def interval(name, start, end, state, regions=None):
    return dict(id=name, frames=[start, end], state=state, description="Synthetic source review",
                evidence_frames=[start], regions=regions or [])


@pytest.fixture
def case(tmp_path):
    obj = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.], [0., 0., 0.]])
    of = np.array([[0, 1, 2], [3, 1, 2]])
    hf = np.array([[0, 2, 1], [0, 3, 2]])
    binding = build_binding(sequence="synthetic", source_video_sha256="f" * 64,
        frame_count=4, fps=24., human_model_type="soma03", human_vertex_count=4,
        human_faces=hf, object_vertices=obj, object_faces=of)
    region = dict(id="candidate", description="Unequal regional support",
                  human_vertex_ids=[0, 1, 2, 3], human_strength=[.8, .7, .5, .4],
                  object_vertex_ids=[0, 1, 2], object_strength=[.7, .7, .7])
    spec = dict(schema="grail.contact_map.v1", binding=binding, reviewed_by="Synthetic fixture",
                hands={"left_hand": [interval("unknown", 0, 1, "unknown"),
                                      interval("touch", 1, 3, "contact", [region]),
                                      interval("release", 3, 4, "no_contact")],
                       "right_hand": [interval("idle", 0, 4, "no_contact")]})
    path = tmp_path / "map.json"
    path.write_text(json.dumps(spec))
    opts = dict(binding=binding, object_vertices=obj, object_faces=of, human_faces=hf,
                dtype=torch.float64, normal_distance_scale_m=0)
    h = torch.tensor([[.003, .003, .024], [.023, .003, .024],
                      [.023, .023, .024], [.003, .023, .024]], dtype=torch.float64)[None].repeat(4, 1, 1)
    r = torch.eye(3, dtype=torch.float64)[None].repeat(4, 1, 1)
    t = torch.zeros(4, 3, dtype=torch.float64)
    return path, spec, opts, h, r, t


def test_live_body_and_object_gradients_and_inactive_frames(case):
    path, _, opts, h, r, t = case
    helper = PhaseContactMapLoss.from_file(path, **opts)
    h.requires_grad_()
    t.requires_grad_()
    report = helper(h, r, t, return_details=True)
    report["total"].backward()
    assert report["total"] > 0 and report["right"] == 0
    assert torch.count_nonzero(h.grad[[0, 3]]) == 0
    assert torch.count_nonzero(t.grad[[0, 3]]) == 0
    assert h.grad[1:3].norm() > 0 and t.grad[1:3].norm() > 0
    torch.testing.assert_close(h.grad.sum((0, 1)), -t.grad.sum(0))
    assert next(iter(report["regions"].values()))["frames"] == [1, 3]


def test_envelope_gradient_matches_finite_difference(case):
    path, _, opts, h, r, t = case
    helper = PhaseContactMapLoss.from_file(path, **opts)
    h.requires_grad_()
    t.requires_grad_()
    assert torch.autograd.gradcheck(lambda x, y: helper(x, r, y), (h, t), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_rotation_gradient_is_live(case):
    path, _, opts, h, _, t = case
    helper = PhaseContactMapLoss.from_file(path, **opts)
    angle = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    def forward(a):
        zero, one = a * 0, a * 0 + 1
        r = torch.stack((torch.stack((a.cos(), zero, a.sin())),
                         torch.stack((zero, one, zero)),
                         torch.stack((-a.sin(), zero, a.cos()))))[None].repeat(4, 1, 1)
        return helper(h, r, t)
    assert torch.autograd.gradcheck(forward, (angle,))
    assert torch.autograd.grad(forward(angle), angle)[0].abs() > 0


@pytest.mark.parametrize("fraction", [.25, .5, 1.])
def test_partial_coverage_caps_mass_and_preserves_normalization(fraction):
    prior = torch.ones(1, 20, dtype=torch.float64) / 20
    cost = torch.arange(20, dtype=torch.float64)[None] * .01
    _, mass = _partial_mass(cost, prior, fraction, 1.6e-5)
    torch.testing.assert_close(mass.sum(-1), torch.ones(1, dtype=torch.float64))
    assert (mass <= prior / fraction + 1e-12).all()
    assert (mass > 1e-6).sum() >= int(20 * fraction)
    if fraction < 1:
        assert (mass < 1e-10).any(), "Partial evidence does not require every colored point"


def test_zero_strength_does_not_acquire_mass():
    prior = torch.tensor([[0., .3, .7]], dtype=torch.float64)
    cost = torch.tensor([[0., .1, .2]], dtype=torch.float64, requires_grad=True)
    loss, mass = _partial_mass(cost, prior, .5, 1.6e-5)
    loss.sum().backward()
    assert mass[0, 0] == 0 and cost.grad[0, 0] == 0
    torch.testing.assert_close(mass.sum(-1), torch.ones(1, dtype=torch.float64))


def test_duplicate_object_seams_do_not_increase_force(case):
    path, spec, opts, h, r, t = case
    original = PhaseContactMapLoss.from_file(path, **opts)
    changed = copy.deepcopy(spec)
    region = changed["hands"]["left_hand"][1]["regions"][0]
    region["object_vertex_ids"].append(3)
    region["object_strength"].append(.7)
    path.write_text(json.dumps(changed))
    duplicated = PhaseContactMapLoss.from_file(path, **opts)
    torch.testing.assert_close(original(h, r, t), duplicated(h, r, t), atol=1e-12, rtol=1e-10)
    assert duplicated.provenance["mass_audit"][0]["object_unique_positions"] == 3


def test_geometry_and_map_are_not_mutated(case):
    path, spec, opts, h, r, t = case
    helper = PhaseContactMapLoss.from_file(path, **opts)
    expected = helper(h, r, t).detach()
    helper.export_spec()["hands"].clear()
    helper.provenance["settings"].clear()
    opts["object_vertices"][:] = 100
    spec["hands"].clear()
    path.write_text("{}")
    torch.testing.assert_close(helper(h, r, t), expected)
    assert helper.provenance["contains_point_correspondences"] is False
    assert helper.export_spec()["schema"] == "grail.contact_map.v1"


def test_additional_same_region_does_not_double_hand_mass(case):
    path, spec, opts, h, r, t = case
    original = PhaseContactMapLoss.from_file(path, **opts)(h, r, t)
    region = copy.deepcopy(spec["hands"]["left_hand"][1]["regions"][0])
    region["id"] = "identical_sampling"
    spec["hands"]["left_hand"][1]["regions"].append(region)
    path.write_text(json.dumps(spec))
    torch.testing.assert_close(PhaseContactMapLoss.from_file(path, **opts)(h, r, t), original)


def test_native_normal_penalty_and_degenerate_safety(case):
    path, _, opts, h, r, t = case
    opts["normal_distance_scale_m"] = .005
    helper = PhaseContactMapLoss.from_file(path, **opts)
    report = helper(h, r, t, return_details=True)
    assert torch.isfinite(report["total"])
    assert (next(iter(report["regions"].values()))["posterior_opposing_cosine"] > .99).all()
    with pytest.raises(ValueError, match="Undefined native surface normals"):
        helper(torch.zeros_like(h), r, t)


def test_reject_scaled_object_rotation_and_changed_native_topology(case):
    path, _, opts, h, r, t = case
    helper = PhaseContactMapLoss.from_file(path, **opts)
    with pytest.raises(ValueError, match="proper rotation"):
        helper(h, r * 2, t)
    with pytest.raises(ValueError, match="topology/timeline"):
        helper(h[:, :3], r, t)
    opts["object_vertices"] *= 2
    with pytest.raises(ValueError, match="geometry/topology"):
        PhaseContactMapLoss.from_file(path, **opts)


@pytest.mark.parametrize("options", [{"partial_mass_fraction": 0}, {"partial_mass_fraction": 1.1},
    {"pair_temperature_m2": 0}, {"coverage_temperature_m2": float("nan")},
    {"normal_distance_scale_m": -1}, {"robust_delta_m": 0}, {"min_opposing_cosine": 2}])
def test_invalid_solver_settings_fail(case, options):
    path, _, opts, _, _, _ = case
    with pytest.raises(ValueError):
        PhaseContactMapLoss.from_file(path, **(opts | options))
