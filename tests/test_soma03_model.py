"""Native integration checks; opt in with GRAIL_GEM_X_SOMA_ASSETS in the pinned runtime."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from grail.models.human_model import create_human_model
from grail.models.soma03_model import (
    IDENTITY_SCHEMA_FIXED_SCALE,
    IDENTITY_SCHEMA_UNIT,
    _sha256,
)


@pytest.fixture(scope="module")
def model():
    assets = os.environ.get("GRAIL_GEM_X_SOMA_ASSETS")
    if not assets:
        pytest.skip("Set GRAIL_GEM_X_SOMA_ASSETS in the isolated SOMA-X 0.3.3 runtime")
    return create_human_model({"body_model": "soma03", "gem_x_soma_assets": assets}, "cpu")


def motion(length=2):
    return dict(
        poses=torch.zeros(length, 77, 3),
        trans=torch.zeros(length, 3),
        identity_coeffs=torch.zeros(length, 128),
        scale_params=torch.ones(length, 57),
        predicted_body_height=1.6967487335205078,
    )


def test_native_geometry_segments_and_observation_routing(model):
    data = motion()
    vertices, faces, joints = model.generate_mesh(data, output_joints=True)
    assert vertices.shape == (2, 4505, 3)
    assert faces.shape == (9006, 3)
    assert joints.shape == (2, 77, 3)
    assert model.get_body_joints(data).shape == (2, 17, 3)
    assert model.get_hand_joints(data).shape == (2, 50, 3)
    left, right = [model.get_segment_indices([s]) for s in ("L_Hand", "R_Hand")]
    assert not np.intersect1d(left, right).size
    assert len(left) == len(right) == 718
    body, hands = model.extract_gt_keypoints(
        dict(vitpose=torch.ones(2, 17, 3), soma_hand_keypoints_2d=torch.zeros(2, 50, 3))
    )
    assert torch.count_nonzero(body[:, [0, 3, 4], 2]) == 0
    assert torch.count_nonzero(body[:, 5:17, 2]) == 24
    assert torch.count_nonzero(hands) == 0
    assert model.get_foot_joint_indices() == (15, 16)


def test_native_toe_contact_regions_are_own_foot_subsets(model):
    left, right = [model.get_segment_indices([name]) for name in ("L_Toe", "R_Toe")]
    assert len(left) > 0 and len(right) > 0
    assert not np.intersect1d(left, right).size
    for side, toes in zip(("L", "R"), (left, right)):
        foot = model.get_segment_indices([side + "_Ankle"])
        assert len(toes) < len(foot)
        assert np.isin(toes, foot).all()


def test_per_frame_identity_is_preserved_not_averaged(model):
    data = motion()
    data["identity_coeffs"][1, 0] = 1.0
    batch = model.generate_mesh(data)[0]
    singles = torch.cat(
        [
            model.generate_mesh(
                {k: v[i : i + 1] if torch.is_tensor(v) else v for k, v in data.items()}
            )[0]
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batch, singles, atol=2e-6, rtol=1e-5)
    assert float((batch[1] - batch[0]).abs().max()) > 1e-4


def test_pose_gradient_reaches_wrist_in_camera_depth(model):
    data = motion(1)
    data["poses"].requires_grad_(True)
    hands = model.get_hand_joints(data, require_grad=True)
    loss = hands[0, 0, 2]
    (gradient,) = torch.autograd.grad(loss, data["poses"])
    assert torch.isfinite(gradient).all()
    # Rotating the left upper arm changes wrist depth while root remains frozen.
    analytic = float(gradient[0, 12, 1])
    assert abs(analytic) > 0.05
    values = []
    for delta in (-1e-3, 1e-3):
        perturbed = motion(1)
        perturbed["poses"][0, 12, 1] = delta
        values.append(float(model.get_hand_joints(perturbed)[0, 0, 2]))
    assert abs(analytic - (values[1] - values[0]) / 2e-3) < 2e-4


@pytest.mark.parametrize("bad", ["legacy", "scaled", "negative"])
def test_rejects_incompatible_shape(model, bad):
    data = motion()
    if bad == "legacy":
        data["identity_coeffs"] = torch.zeros(2, 45)
    elif bad == "scaled":
        data["scale_params"][:, 0] = 0.9
    else:
        data["scale_params"][:, 1] = -1
    with pytest.raises(ValueError):
        model.generate_mesh(data)


def test_transform_places_actual_root_and_preserves_local_pose(model):
    incam = motion()
    incam["trans"][:] = torch.tensor([0.2, -0.8, 3.0])
    incam["poses"][:, 15, 0] = 0.2
    camera_rotation = axis_angle_to_matrix(torch.tensor([0.2, -0.3, 0.4]))
    camera_translation = torch.tensor([0.4, 0.7, 0.2])
    original = model._forward(incam, joints_only=True)["joints"][:, 0]
    identity = (torch.zeros(1, 128), torch.ones(1, 57))
    identity[0][0, 0] = 0.3
    transformed = model.transform_global_motion(
        incam,
        incam,
        cam_R=camera_rotation,
        cam_t=camera_translation,
        gt_shape_params=identity,
        gt_height=123.0,
        device="cpu",
    )
    ratio = model.compute_tpose_height(identity) / incam["predicted_body_height"]
    expected = camera_translation + ratio * (original @ camera_rotation.T)
    actual = model._forward(transformed, joints_only=True)["joints"][:, 0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(transformed["poses"][:, 1:], incam["poses"][:, 1:])
    assert torch.equal(transformed["scale_params"][:, 0], torch.ones(2))
    assert incam["poses"][0, 0].count_nonzero() == 0
    with pytest.raises(ValueError, match="twice"):
        model.transform_global_motion(
            transformed, transformed, cam_R=camera_rotation, cam_t=camera_translation
        )


def test_global_alignment_matches_incam_root_at_anchor(model):
    incam, globally = motion(), motion()
    incam["trans"] += torch.tensor([0.1, 0.2, 3.0])
    incam["poses"][:, 0, 1] = 0.4
    globally["trans"] += torch.tensor([1.0, 2.0, 0.0])
    globally["poses"][:, 0, 1] = -0.7
    output = model.transform_global_motion(
        globally,
        incam,
        cam_R=torch.eye(3),
        cam_t=torch.zeros(3),
        align_frame=1,
        use_global=True,
        device="cpu",
    )
    torch.testing.assert_close(
        model._forward(output, joints_only=True)["joints"][1, 0],
        model._forward(incam, joints_only=True)["joints"][1, 0],
        atol=2e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        axis_angle_to_matrix(output["poses"][1, 0]), axis_angle_to_matrix(incam["poses"][1, 0])
    )


def test_fixed_identity_is_bound_to_parameters_and_native_model(model, tmp_path):
    original = Path(os.environ.get(
        "GRAIL_SOMA_UNIT_IDENTITY",
        Path(__file__).resolve().parents[1] / "data/body_profiles/unitree_g1/soma/soma03_g1_identity.npz",
    ))
    if not original.exists():
        pytest.skip("Fit or install the G1-like identity before the native identity check")
    path = tmp_path / original.name
    shutil.copyfile(original, path)
    shutil.copyfile(original.with_suffix(".json"), path.with_suffix(".json"))
    cfg = {"soma_identity_params_path": str(path)}
    identity, scales = model.load_shape_params(cfg, "unused", "cpu")
    assert identity.shape == (1, 128) and scales.shape == (1, 57)
    assert scales[0, 0] == 1
    metadata = json.loads(path.with_suffix(".json").read_text())
    for key in metadata["sources"]:
        if key.endswith("assets/soma03/SOMA_neutral.npz"):
            metadata["sources"][key] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model binding"):
        model.load_shape_params(cfg, "unused", "cpu")
    shutil.copyfile(original.with_suffix(".json"), path.with_suffix(".json"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        model.load_shape_params(cfg, "unused", "cpu")


@pytest.fixture
def scaling_model(model):
    """Keep these authorization tests independent of the shared native layer."""
    previous_scale = model._fixed_identity_global_scale
    previous_validation = model.model.validate_external_global_scale
    previous_provenance = getattr(model, "identity_provenance", None)
    model._fixed_identity_global_scale = None
    model.model.validate_external_global_scale = True
    yield model
    model._fixed_identity_global_scale = previous_scale
    model.model.validate_external_global_scale = previous_validation
    if previous_provenance is None:
        model.__dict__.pop("identity_provenance", None)
    else:
        model.identity_provenance = previous_provenance


@pytest.fixture(scope="module")
def native_bindings(model):
    return {
        "/original/" + suffix: digest
        for suffix, digest in model.model.compatibility["legacy_identity_bindings"].items()
    }


def write_identity(path, native_bindings, global_scale=0.8, schema=IDENTITY_SCHEMA_FIXED_SCALE):
    identity = np.zeros((1, 128), np.float32)
    identity[0, 0] = 0.25
    scales = np.ones((1, 57), np.float32)
    scales[:, 0] = global_scale
    np.savez(path, identity_coeffs=identity, scale_params=scales, schema=np.asarray(schema))
    metadata = {
        "schema": schema,
        "global_scale": float(scales[0, 0]),
        "global_scale_policy": "fixed_identity_parameter",
        "artifacts": {path.name: _sha256(path)},
        "sources": native_bindings.copy(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return metadata


def test_only_manifest_authorizes_one_constant_identity_scale(
    scaling_model, native_bindings, tmp_path
):
    model = scaling_model
    data = motion()
    data["scale_params"][:, 0] = 0.8
    with pytest.raises(ValueError, match="validated v2 manifest"):
        model.generate_mesh(data)
    path = tmp_path / "fixed.npz"
    write_identity(path, native_bindings)
    identity, scales = model.load_shape_params({"soma_identity_params_path": path}, "unused", "cpu")
    assert model.model.validate_external_global_scale is False
    data["identity_coeffs"], data["scale_params"] = identity, scales
    assert model.generate_mesh(data)[0].shape == (2, 4505, 3)
    assert model.generate_mesh(motion())[0].shape == (2, 4505, 3)
    for channels in ([0.8, 1.0], [0.8, 0.7], [0.7, 0.7]):
        invalid = motion()
        invalid["scale_params"][:, 0] = torch.tensor(channels)
        with pytest.raises(ValueError, match="constant fixed identity scale"):
            model.generate_mesh(invalid)
    data["scale"] = 0.8
    with pytest.raises(ValueError, match="extra mesh scale"):
        model.generate_mesh(data)
    unit_path = tmp_path / "unit.npz"
    write_identity(unit_path, native_bindings, 1.0, IDENTITY_SCHEMA_UNIT)
    model.load_shape_params({"soma_identity_params_path": unit_path}, "unused", "cpu")
    assert model._fixed_identity_global_scale is None
    assert model.model.validate_external_global_scale is True
    data.pop("scale")
    with pytest.raises(ValueError, match="validated v2 manifest"):
        model.generate_mesh(data)


@pytest.mark.parametrize(
    "tamper", ["checksum", "binding", "policy", "zero", "nan", "npz_schema", "npz_scale"]
)
def test_failed_identity_load_never_changes_authorization(
    scaling_model, native_bindings, tmp_path, tamper
):
    model = scaling_model
    good = tmp_path / "good.npz"
    write_identity(good, native_bindings)
    bad = tmp_path / "bad.npz"
    metadata = write_identity(bad, native_bindings, 0.7)
    if tamper == "checksum":
        metadata["artifacts"][bad.name] = "0" * 64
    elif tamper == "binding":
        metadata["sources"][next(iter(native_bindings))] = "0" * 64
    elif tamper == "policy":
        metadata["global_scale_policy"] = "fit_per_frame"
    elif tamper in ("zero", "nan"):
        metadata["global_scale"] = 0.0 if tamper == "zero" else float("nan")
    else:
        with np.load(bad, allow_pickle=False) as original:
            arrays = {key: original[key] for key in original.files}
        if tamper == "npz_schema":
            arrays["schema"] = np.asarray(IDENTITY_SCHEMA_UNIT)
        else:
            arrays["scale_params"][:, 0] = 1.0
        np.savez(bad, **arrays)
        metadata["artifacts"][bad.name] = _sha256(bad)
    bad.with_suffix(".json").write_text(json.dumps(metadata))
    for already_authorized in (False, True):
        if already_authorized:
            model.load_shape_params({"soma_identity_params_path": good}, "unused", "cpu")
        previous = (
            model._fixed_identity_global_scale,
            model.model.validate_external_global_scale,
            getattr(model, "identity_provenance", None),
        )
        with pytest.raises(ValueError):
            model.load_shape_params({"soma_identity_params_path": bad}, "unused", "cpu")
        assert previous == (
            model._fixed_identity_global_scale,
            model.model.validate_external_global_scale,
            getattr(model, "identity_provenance", None),
        )


def test_fixed_scale_uses_native_geometry_once_and_preserves_frames(
    scaling_model, native_bindings, tmp_path
):
    model = scaling_model
    path = tmp_path / "fixed.npz"
    write_identity(path, native_bindings)
    identity, scales = model.load_shape_params({"soma_identity_params_path": path}, "unused", "cpu")
    data = motion(3)
    data["poses"][:, 12, 1] = torch.tensor([0.1, 0.3, -0.2])
    data["poses"][:, 15, 0] = torch.tensor([0.05, -0.1, 0.15])
    data["trans"][:] = torch.tensor([0.2, -0.4, 2.0])
    data["identity_coeffs"] = identity.expand(3, -1).clone()
    # Static batching must preserve each shape and fixed GS; temporal reduction
    # in the released GEM wrapper would average identity and drop the scale.
    data["identity_coeffs"][1, 1] = 0.3
    data["scale_params"] = scales.expand(3, -1).clone()
    vertices, _, joints = model.generate_mesh(data, output_joints=True)
    native = model.model.soma
    native.prepare_identity(
        data["identity_coeffs"],
        data["scale_params"][:, 1:],
        global_scale=data["scale_params"][:, :1],
        repose_to_bind_pose=True,
    )
    direct = native.pose(data["poses"], transl=data["trans"], apply_correctives=True, fk_only=False)
    torch.testing.assert_close(vertices, direct["vertices"], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(joints, direct["joints"], atol=2e-6, rtol=1e-5)
    singles = torch.cat(
        [
            model.generate_mesh(
                {
                    key: value[i : i + 1] if torch.is_tensor(value) else value
                    for key, value in data.items()
                }
            )[0]
            for i in range(3)
        ]
    )
    torch.testing.assert_close(vertices, singles, atol=2e-6, rtol=1e-5)
    assert (vertices[1] - vertices[0]).abs().max() > 1e-4

    source = motion(3)
    source["poses"] = data["poses"].clone()
    source["trans"] = data["trans"].clone()
    source["predicted_body_height"] = model.compute_tpose_height(source)
    before_pose = source["poses"].clone()
    before_scales = source["scale_params"].clone()
    source_roots = model._forward(source, joints_only=True)["joints"][:, 0].clone()
    output = model.transform_global_motion(
        source,
        source,
        cam_R=torch.eye(3),
        cam_t=torch.zeros(3),
        gt_shape_params=(identity, scales),
        device="cpu",
    )
    expected_ratio = (
        model.compute_tpose_height((identity, scales)) / source["predicted_body_height"]
    )
    actual_roots = model._forward(output, joints_only=True)["joints"][:, 0]
    torch.testing.assert_close(actual_roots, source_roots * expected_ratio, atol=2e-6, rtol=1e-5)
    assert torch.equal(output["poses"], before_pose)
    assert torch.equal(source["poses"], before_pose)
    assert torch.equal(source["scale_params"], before_scales)
    assert output["scale"] == 1.0
    assert torch.equal(output["scale_params"][:, 0], scales[0, 0].expand(3))
    assert output["soma03_placement"]["external_global_scale"] == float(scales[0, 0])
    assert output["soma03_placement"]["global_scale_policy"] == "fixed_identity_parameter"
