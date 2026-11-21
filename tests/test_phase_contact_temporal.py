"""Material temporal controls use live geometry and explicit source intervals."""

import copy
import json

import numpy as np
import pytest
import torch

from grail.optimization.fixed_contact import build_binding
from grail.optimization.phase_contact_map_loss import PhaseContactMapLoss
from grail.optimization.phase_contact_temporal import PhaseContactTemporalLoss, _window_envelope


@pytest.fixture
def case(tmp_path):
    def build(fps=24.):
        vertices = np.array([[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.], [.2, .2, 0.]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        binding = build_binding(sequence="temporal", source_video_sha256="f" * 64,
            frame_count=8, fps=fps, human_model_type="soma03", human_vertex_count=4,
            human_faces=faces, object_vertices=vertices, object_faces=faces)
        region = dict(id="patch", description="Source authored material region",
                      human_vertex_ids=[0, 1, 2, 3], human_strength=[1., .5, .8, 0.],
                      object_vertex_ids=[0, 1, 2], object_strength=[1., .8, .8])
        def interval(name, start, end, state):
            return dict(id=name, frames=[start, end], state=state,
                description="Observed phase", evidence_frames=[start],
                regions=[region] if state == "contact" else [])
        spec = dict(schema="grail.contact_map.v1", binding=binding, reviewed_by="Test fixture",
            hands={"left_hand": [interval("approach", 0, 2, "no_contact"),
                interval("acquire", 2, 4, "contact"), interval("hold", 4, 7, "contact"),
                interval("release", 7, 8, "no_contact")],
                "right_hand": [interval("idle", 0, 8, "no_contact")]})
        path = tmp_path / "map.json"
        path.write_text(json.dumps(spec))
        contact = PhaseContactMapLoss.from_file(path, binding=binding,
            object_vertices=vertices, object_faces=faces, human_faces=faces, dtype=torch.float64)
        h = torch.tensor(vertices, dtype=torch.float64)[None].repeat(8, 1, 1)
        r = torch.eye(3, dtype=torch.float64)[None].repeat(8, 1, 1)
        t = torch.zeros(8, 3, dtype=torch.float64)
        return contact, h, r, t
    return build


def helper(contact, **overrides):
    options = dict(fps=contact.export_spec()["binding"]["fps"],
        hold_interval_ids={"left": ["hold"], "right": []},
        windows=[dict(id="onset", frames=[1, 7], regions=["left/hold/patch"], object=True)],
        hold_speed_tolerance_mps=0., hold_speed_scale_mps=1.,
        window_speed_tolerance_mps=0., window_speed_scale_mps=1.,
        window_acceleration_tolerance_mps2=0., window_acceleration_scale_mps2=1.)
    options.update(overrides)
    return PhaseContactTemporalLoss(contact, **options)


def rotation_z(angle):
    zero, one = angle * 0, angle * 0 + 1
    return torch.stack((torch.stack((angle.cos(), -angle.sin(), zero), -1),
                        torch.stack((angle.sin(), angle.cos(), zero), -1),
                        torch.stack((zero, zero, one), -1)), -2)


def test_rigid_co_motion_cancels_world_translation_and_full_rotation(case):
    contact, h, _, t = case()
    r = rotation_z(torch.linspace(0, .8, 8, dtype=h.dtype))
    t[:, 0] = torch.linspace(0, 1, 8)
    h = h @ r.transpose(-1, -2) + t[:, None]
    out = helper(contact)(h, r, t)
    assert out["hold_relative_velocity"].abs() < 1e-25
    assert out["window_velocity"] > 0
    assert out["window_acceleration"] > 0


def test_no_hold_constraint_bridges_acquisition_or_release(case):
    contact, h, r, t = case()
    h[:4] += 100
    h[7:] -= 100
    h.requires_grad_()
    t.requires_grad_()
    loss = helper(contact, windows=[])(h, r, t)["hold_relative_velocity"]
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(h.grad) == 0
    assert torch.count_nonzero(t.grad) == 0
    h2 = h.detach().clone()
    h2[5, 0, 0] += .1
    assert helper(contact, windows=[])(h2, r, t)["hold_relative_velocity"] > 0


def test_constant_speed_has_zero_acceleration_and_no_outside_window_gradient(case):
    contact, h, r, t = case()
    translation = torch.arange(8, dtype=h.dtype)[:, None, None] * torch.tensor([.02, .03, 0.], dtype=h.dtype)
    h = (h + translation).requires_grad_()
    t = translation[:, 0].clone().requires_grad_()
    temporal = helper(contact, hold_interval_ids={"left": [], "right": []})
    out = temporal(h, r, t)
    assert out["window_acceleration"] < 1e-24
    out["window_velocity"].backward()
    assert torch.count_nonzero(h.grad[[0, 7]]) == 0
    assert torch.count_nonzero(t.grad[[0, 7]]) == 0
    assert h.grad[1:7].norm() > 0


def test_temporal_units_follow_source_fps(case):
    contact24, h, r, t = case(24.)
    contact48, _, _, _ = case(48.)
    h[5, 0, 0] += .1
    a, b = helper(contact24)(h, r, t), helper(contact48)(h, r, t)
    for key in ("hold_relative_velocity", "window_velocity"):
        torch.testing.assert_close(b[key], 4 * a[key])
    torch.testing.assert_close(b["window_acceleration"], 16 * a["window_acceleration"])
    with pytest.raises(ValueError, match="bound source timeline"):
        helper(contact24, fps=48.)


def test_rotation_and_human_object_translation_have_live_correct_gradients(case):
    contact, h, _, t = case()
    torch.manual_seed(6)
    h = (h + torch.randn_like(h) * .01).requires_grad_()
    t = (t + torch.randn_like(t) * .02).requires_grad_()
    angles = torch.linspace(.01, .2, 8, dtype=h.dtype).requires_grad_()
    temporal = helper(contact)
    def run(x, y, a):
        return sum(temporal(x, rotation_z(a), y).values())
    assert torch.autograd.gradcheck(run, (h, t, angles), eps=1e-6, atol=2e-5, rtol=1e-4)
    grads = torch.autograd.grad(run(h, t, angles), (h, t, angles))
    assert all(g.norm() > 0 for g in grads)


def test_material_rotation_cannot_hide_in_stationary_centroid(case):
    contact, h, r, t = case()
    mass = contact.regions[1].human_mass
    center = (h * mass[None, :, None]).sum(1, keepdim=True)
    moving = (h - center) @ rotation_z(torch.linspace(0, .8, 8, dtype=h.dtype)).transpose(-1, -2) + center
    torch.testing.assert_close((moving * mass[None, :, None]).sum(1), center[:, 0])
    assert helper(contact)(moving, r, t)["hold_relative_velocity"] > 0


def test_authored_weights_are_fixed_and_inputs_are_not_mutated(case):
    contact, h, r, t = case()
    spec = copy.deepcopy(contact.export_spec())
    temporal = helper(contact)
    h[5, 3] += 100  # Zero authored strength stays zero in temporal terms.
    out = temporal(h, r, t)
    assert all(value == 0 for value in out.values())
    assert spec == contact.export_spec()
    temporal.provenance["settings"].clear()
    assert temporal.provenance["settings"]["fps"] == 24.
    torch.testing.assert_close(temporal.object_mass, torch.tensor([1, 2, 2, 1], dtype=h.dtype) / 6)


def test_speed_hinge_and_optional_ramp(case):
    contact, h, r, t = case()
    h[5:] += .001
    original = helper(contact, windows=[])(h, r, t)["hold_relative_velocity"]
    ramp = helper(contact, windows=[], hold_ramp_frames=4)(h, r, t)["hold_relative_velocity"]
    assert 0 < ramp < original
    assert helper(contact, windows=[], hold_speed_tolerance_mps=.1)(h, r, t)["hold_relative_velocity"] == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"hold_interval_ids": {"left": ["missing"], "right": []}}, "name contact"),
    ({"hold_interval_ids": {"left": ["release"], "right": []}}, "name contact"),
    ({"hold_interval_ids": {"left": ["hold"]}}, "left and right"),
    ({"windows": [dict(id="bad", frames=[-1, 5], regions=[], object=True)]}, "half-open"),
    ({"windows": [dict(id="bad", frames=[1, 9], regions=[], object=True)]}, "half-open"),
    ({"windows": [dict(id="bad", frames=[1, 3], regions=[], object=True)]}, "half-open"),
    ({"windows": [dict(id="bad", frames=[1, 5], regions=["left/missing/patch"], object=True)]}, "bound region"),
    ({"windows": [dict(id="bad", frames=[1, 5], regions=[], object=False)]}, "select"),
    ({"window_speed_scale_mps": 0.}, "positive"),
    ({"hold_ramp_frames": 1.5}, "integer"),
])
def test_invalid_scope_and_units_are_rejected(case, kwargs, message):
    contact, _, _, _ = case()
    with pytest.raises(ValueError, match=message):
        helper(contact, **kwargs)


def test_invalid_geometry_and_empty_scopes(case):
    contact, h, r, t = case()
    temporal = helper(contact, windows=[], hold_interval_ids={"left": [], "right": []})
    out = temporal(h.requires_grad_(), r, t)
    sum(out.values()).backward()
    assert torch.count_nonzero(h.grad) == 0
    with pytest.raises(ValueError, match="proper rotation"):
        temporal(h, r * 2, t)
    with pytest.raises(ValueError, match="topology/timeline"):
        temporal(h[:, :3], r, t)


def test_window_defaults_preserve_unweighted_hard_window(case):
    contact, h, r, t = case()
    h[4, 0, 0] += .03
    implicit = helper(contact)(h, r, t)
    explicit = helper(contact, windows=[dict(id="onset", frames=[1, 7],
        regions=["left/hold/patch"], object=True, ramp_frames=0,
        velocity_weight=1., acceleration_weight=1.)])(h, r, t)
    for key in implicit:
        torch.testing.assert_close(implicit[key], explicit[key], rtol=0, atol=0)


def test_quintic_window_envelope_uses_minimum_stencil_weights():
    frame, edge, triple = _window_envelope(9, 2, dtype=torch.float64)
    torch.testing.assert_close(frame, torch.tensor([0., .5, 1., 1., 1., 1., 1., .5, 0.], dtype=frame.dtype))
    torch.testing.assert_close(edge, torch.minimum(frame[:-1], frame[1:]))
    torch.testing.assert_close(triple, torch.minimum(frame[:-2], torch.minimum(frame[1:-1], frame[2:])))
    assert edge.sum() < len(edge) and triple.sum() < len(triple)


def test_feathered_window_has_no_boundary_stencil_gradient_leak(case):
    contact, h, r, t = case()
    h[1, 0] += 10
    h[6, 0] -= 10
    h.requires_grad_()
    temporal = helper(contact, hold_interval_ids={"left": [], "right": []},
        windows=[dict(id="soft", frames=[1, 7], regions=["left/hold/patch"],
                      object=True, ramp_frames=2)])
    output = temporal(h, r, t, return_details=True)
    loss = output["window_velocity"] + output["window_acceleration"]
    assert loss == 0  # The endpoint-containing derivatives have zero weight.
    loss.backward()
    assert torch.count_nonzero(h.grad) == 0
    assert output["details"]["windows"]["soft"]["frame_envelope"][[0, -1]].sum() == 0


def test_weak_background_acceleration_covers_motion_outside_local_window(case):
    contact, h, r, t = case()
    h[0, 0, 0] += .1
    local = dict(id="local", frames=[2, 7], regions=["left/hold/patch"], object=True, ramp_frames=1)
    background = dict(id="background", frames=[0, 8], regions=["left/hold/patch"],
                      object=True, ramp_frames=0, velocity_weight=0., acceleration_weight=.1)
    local_only = helper(contact, windows=[local])(h, r, t)
    background_only = helper(contact, windows=[background])(h, r, t)
    both_helper = helper(contact, windows=[local, background])
    both = both_helper(h, r, t)
    assert local_only["window_acceleration"] == 0
    assert both["window_acceleration"] > 0 and both["window_velocity"] == 0
    # Arithmetic mean remains explicit; small background weight is not normalized away.
    torch.testing.assert_close(both["window_acceleration"], background_only["window_acceleration"] / 2)
    mass = both_helper.provenance["window_weights"][1]
    assert sum(mass["effective_velocity_edge_weights"]) == 0
    assert sum(mass["effective_acceleration_triple_weights"]) == pytest.approx(.05)


def test_fade_mass_is_not_renormalized_and_coefficients_are_independent(case):
    contact, h, r, t = case()
    # Constant nonzero material velocity: exact contribution follows fade mass.
    h = h + torch.arange(8, dtype=h.dtype)[:, None, None] * .01
    base_window = dict(id="local", frames=[0, 8], regions=["left/hold/patch"], object=False)
    baseline = helper(contact, windows=[base_window])(h, r, t)["window_velocity"]
    temporal = helper(contact, windows=[dict(base_window, ramp_frames=2,
        velocity_weight=.4, acceleration_weight=.2)])
    current = temporal(h, r, t)["window_velocity"]
    effective = temporal.provenance["window_weights"][0]["effective_velocity_edge_weights"]
    torch.testing.assert_close(current, baseline * sum(effective))
    assert current < baseline * .4


@pytest.mark.parametrize("options", [
    dict(ramp_frames=-1), dict(ramp_frames=1.5), dict(velocity_weight=-1.),
    dict(acceleration_weight=float("nan")), dict(velocity_weight=0., acceleration_weight=0.),
])
def test_invalid_window_fade_and_weights(case, options):
    contact, _, _, _ = case()
    window = dict(id="invalid", frames=[0, 8], regions=["left/hold/patch"], object=True, **options)
    with pytest.raises(ValueError):
        helper(contact, windows=[window])
