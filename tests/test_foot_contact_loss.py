from types import SimpleNamespace

import pytest
import torch

from grail.optimization.loss_computer import LossComputer


class MockHumanModel:
    _SEGMENT_INDEX = {"L_Ankle": 0, "L_Toe": 1, "R_Ankle": 2, "R_Toe": 3}

    def get_foot_joint_indices(self):
        return 0, 1

    def get_verts_segment(self, verts, labels):
        return verts[:, self._SEGMENT_INDEX[labels[0]] : self._SEGMENT_INDEX[labels[0]] + 1]


def make_loss_computer():
    return LossComputer(None, MockHumanModel(), "cpu", 2, None)


def make_inputs(joints, contacts, verts=None):
    if verts is None:
        verts = torch.zeros(len(joints), 4, 3)
    data = SimpleNamespace(human=SimpleNamespace(foot_contact_probs=contacts))
    pred = SimpleNamespace(human=SimpleNamespace(body_joints_seq=joints, verts_seq=verts))
    return data, pred


def test_default_matches_legacy_hard_3d_ankle_loss():
    joints = torch.zeros(3, 2, 3)
    joints[:, 0, 0] = torch.tensor([0.0, 0.1, 0.3])
    contacts = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    data, pred = make_inputs(joints, contacts)

    loss = make_loss_computer()._human_foot_contact_loss(data, pred, {}, weight=2.0)

    assert loss.item() == pytest.approx(0.3)


def test_soft_planar_loss_ignores_vertical_motion():
    joints = torch.zeros(3, 2, 3)
    joints[:, 0, 0] = torch.tensor([0.0, 0.1, 0.2])
    joints[:, 0, 2] = torch.tensor([0.0, 1.0, 2.0])
    contacts = torch.tensor([[0.8, 0.0, 0.0, 0.0]]).repeat(3, 1)
    data, pred = make_inputs(joints, contacts)

    cfg = {"weight_mode": "soft", "soft_floor": 0.0, "velocity_space": "planar"}
    loss = make_loss_computer()._human_foot_contact_loss(data, pred, cfg, weight=1.0)

    assert loss.item() == pytest.approx(0.1)


def test_vertical_weight_reduces_vertical_contact_penalty():
    joints = torch.zeros(3, 2, 3)
    joints[:, 0, 2] = torch.tensor([0.0, 1.0, 2.0])
    contacts = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    data, pred = make_inputs(joints, contacts)

    loss = make_loss_computer()._human_foot_contact_loss(
        data, pred, {"vertical_weight": 0.25}, weight=1.0
    )

    assert loss.item() == pytest.approx(0.25)


def test_foot_parts_long_horizon_penalizes_stance_drift():
    joints = torch.zeros(3, 2, 3)
    verts = torch.zeros(3, 4, 3)
    verts[:, :, 0] = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1)
    contacts = torch.ones(3, 4)
    data, pred = make_inputs(joints, contacts, verts=verts)

    cfg = {
        "point_mode": "foot_parts",
        "velocity_space": "planar",
        "long_horizon_lag": 2,
        "long_horizon_weight": 0.5,
    }
    loss = make_loss_computer()._human_foot_contact_loss(data, pred, cfg, weight=1.0)

    assert loss.item() == pytest.approx(2.0)


def test_stance_term_reaches_foot_mesh_controls_and_leaves_swing_free():
    joints = torch.zeros(6, 2, 3)
    verts = torch.zeros(6, 4, 3)
    verts[:, 1, 0] = torch.arange(6) * 0.02
    verts.requires_grad_(True)
    contacts = torch.zeros(6, 4)
    contacts[:, 1] = 1.0
    data, pred = make_inputs(joints, contacts, verts)
    computer = make_loss_computer()
    cfg = {"point_mode": "foot_parts", "velocity_space": "planar"}
    legacy = computer._human_foot_contact_loss(data, pred, cfg, weight=10.0)
    candidate = computer._human_foot_contact_loss(
        data, pred, {**cfg, "stance_weight": 2.0}, weight=10.0
    )
    assert candidate > legacy
    candidate.backward()
    assert torch.isfinite(verts.grad).all()
    assert verts.grad[:, 1, :2].abs().sum() > 0
    assert torch.count_nonzero(verts.grad[:, [0, 2, 3]]) == 0


def test_soma_manipulation_enables_support_drift_in_both_fitting_stages():
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/recon_4dhoi/manip_soma.yaml").read_text()
    )
    stages = config["optimization"]["opt_stage_specs"]
    for name in ("init_opt", "contact_opt"):
        foot = stages[name]["loss_cfg"]["human_foot_contact"]
        assert foot["weight"] > 0 and foot["stance_weight"] > 0
        assert foot["point_mode"] == "foot_parts"
        assert foot["velocity_space"] == "planar"
        assert foot["long_horizon_lag"] > 1 and foot["long_horizon_weight"] > 0
