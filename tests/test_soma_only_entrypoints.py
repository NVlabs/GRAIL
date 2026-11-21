"""Retired body models must fail before estimation, even when reusing predictions."""

import sys

import pytest

from grail.core.config import DEFAULT_RECON_CONFIG, load_recon_config


@pytest.mark.parametrize("tag", ["smplx", "g1_smplx"])
def test_reconstruction_rejects_retired_model_with_step1_skipped(tag, tmp_path, monkeypatch):
    import yaml
    from grail.pipelines import recon_4dhoi

    config, _ = load_recon_config(DEFAULT_RECON_CONFIG)
    config["human_model"]["body_model"] = tag
    config_path = tmp_path / "case.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["recon_4dhoi", "--config", str(config_path), "--skip_step1"])
    monkeypatch.setattr(recon_4dhoi, "glob", lambda *_: pytest.fail("unexpected cache discovery"))
    with pytest.raises(SystemExit) as error:
        recon_4dhoi.main()
    assert error.value.code == 2


@pytest.mark.parametrize("tag", ["smplx", "g1_smplx"])
def test_retired_models_rejected_before_model_or_motion_assets_are_opened(tag):
    from grail.core.io import load_human_motion_data
    from grail.models.human_model import create_human_model
    from grail.pose_est.human_pose import run_human_pose_est

    with pytest.raises(ValueError, match="body_model"):
        create_human_model({"body_model": tag}, "cpu")
    with pytest.raises(ValueError, match="body model"):
        run_human_pose_est("unused.mp4", model=tag)
    with pytest.raises(ValueError, match="motion model"):
        load_human_motion_data("unused.npz", model=tag)
