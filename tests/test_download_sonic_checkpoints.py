import os
from pathlib import Path
import shutil
import subprocess


def test_install_sonic_warmstarts_and_restore_missing_config(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/setup/download_checkpoints.sh"
    installer = tmp_path / "scripts/setup" / script.name
    installer.parent.mkdir(parents=True)
    shutil.copyfile(script, installer)
    # Record the requested remote filename in each installed file.
    (tmp_path / "huggingface_hub.py").write_text(
        "from pathlib import Path\n"
        "def hf_hub_download(*, filename, **kwargs):\n"
        "    cache = Path('cache') / filename\n"
        "    cache.parent.mkdir(parents=True, exist_ok=True)\n"
        "    cache.write_text(filename)\n"
        "    return cache\n"
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    command = ["bash", str(installer), "--skip-gem-x", "--skip-mose3",
               "--skip-foundationpose", "--skip-hunyuan3d"]
    subprocess.run(command, check=True, env=env, capture_output=True)
    models = tmp_path / "imports/SONIC/models"
    for name in ("pnp_table", "pnp_ground", "terrain_stairs"):
        for file in ("last.pt", "config.yaml"):
            assert (models / name / file).read_text() == f"checkpoint/SONIC/models/{name}_warmstart/{file}"
    for name, file in (("manip_warmstart", "last.pt"), ("manip_warmstart", "meta.json"),
                       ("sonic_manipulation_base", "last.pt"), ("sonic_manipulation_base", "model_config.yaml")):
        assert (models / name / file).read_text() == f"checkpoint/SONIC/models/{name}/{file}"

    config = models / "terrain_stairs/config.yaml"
    config.unlink()
    subprocess.run(command, check=True, env=env, capture_output=True)
    assert config.read_text() == "checkpoint/SONIC/models/terrain_stairs_warmstart/config.yaml"
