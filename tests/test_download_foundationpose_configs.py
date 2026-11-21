import os
from pathlib import Path
import shutil
import subprocess


def test_restore_predictor_configs_for_existing_weights(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/setup/download_checkpoints.sh"
    installer = tmp_path / "scripts/setup" / script.name
    installer.parent.mkdir(parents=True)
    shutil.copyfile(script, installer)
    (tmp_path / "huggingface_hub.py").write_text(
        "from pathlib import Path\n"
        "def hf_hub_download(*, filename, **kwargs):\n"
        "    assert filename.endswith('/config.yml'), filename\n"
        "    cache = Path('cache') / filename\n"
        "    cache.parent.mkdir(parents=True, exist_ok=True)\n"
        "    cache.write_text(filename)\n"
        "    return cache\n"
    )
    weights = tmp_path / "imports/FoundationPose/weights"
    predictors = ("2024-01-11-20-02-45", "2023-10-28-18-33-37")
    for name in predictors:
        (weights / name).mkdir(parents=True)
        (weights / name / "model_best.pth").write_bytes(b"existing checkpoint")
    command = [
        "bash", str(installer), "--skip-gem-x", "--skip-mose3",
        "--skip-hunyuan3d", "--skip-sonic", "--skip-sonic-manip",
    ]
    subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(tmp_path)), check=True, capture_output=True)
    for name in predictors:
        assert (weights / name / "model_best.pth").read_bytes() == b"existing checkpoint"
        assert (weights / name / "config.yml").read_text() == f"checkpoint/FoundationPose/weights/{name}/config.yml"
