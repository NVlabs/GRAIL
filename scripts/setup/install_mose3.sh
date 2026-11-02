#!/usr/bin/env bash
# Install the pinned MoSE3 worker without changing GRAIL's PyTorch runtime.
# Usage: bash scripts/setup/install_mose3.sh [--python PYTHON] [--venv DIRECTORY] [--source DIRECTORY]
# Weights are separate: hf download JoannaCCCCCC/MoSE3 --local-dir imports/MoSE3/ckpt
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${GRAIL_BASE_PYTHON:-python}"
VENV_DIR="$PROJECT_ROOT/.venvs/mose3"
SOURCE_DIR="$PROJECT_ROOT/imports/MoSE3"
SOURCE_REVISION=59807f07ff27b5afdf4f35742ac7be88d25e0e3a

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python|--venv|--source)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --python) BASE_PYTHON="$2" ;;
                --venv) VENV_DIR="$2" ;;
                --source) SOURCE_DIR="$2" ;;
            esac
            shift 2 ;;
        --help|-h) sed -n '2,4s/^# //p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
BASE_PYTHON="$("$BASE_PYTHON" -c 'import sys; print(sys.executable)')"
VENV_DIR="$("$BASE_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).absolute())' "$VENV_DIR")"
[ -f "$SOURCE_DIR/mose3/models/mose3.py" ] || {
    echo "MoSE3 source missing. Run git submodule update --init imports/MoSE3." >&2; exit 1;
}
[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$SOURCE_REVISION" ] || {
    echo "Expected MoSE3 at $SOURCE_REVISION; update the submodule before installing." >&2; exit 1;
}
[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ] || {
    echo "MoSE3 tracked source has local edits; preserve it and use the pinned submodule." >&2; exit 1;
}

"$BASE_PYTHON" - "$VENV_DIR" "$SOURCE_REVISION" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit('MoSE3 requires Python >= 3.10.')
destination = Path(sys.argv[1])
marker = destination / '.grail-mose3-setup.json'
identity = {'source_revision': sys.argv[2], 'torch': '2.6.0+cu126'}
if destination.exists():
    if not marker.is_file() or json.loads(marker.read_text()) != identity:
        raise SystemExit(f'Refusing to modify an unowned/differently configured environment: {destination}')
else:
    subprocess.run([sys.executable, '-m', 'venv', str(destination)], check=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
if 'include-system-site-packages = false' not in (destination / 'pyvenv.cfg').read_text().lower():
    raise SystemExit('MoSE3 must use an isolated environment without system-site packages.')
PY

WORKER_PYTHON="$VENV_DIR/bin/python"
export PIP_REQUIRE_VIRTUALENV=true
# The base Python seeds pip23, whose wheel-name normalization rejects current
# typing_extensions metadata on the CUDA index. Upgrade before resolving Torch.
"$WORKER_PYTHON" -m pip install 'pip==26.2.2' 'setuptools==79.0.1' wheel
"$WORKER_PYTHON" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu126
"$WORKER_PYTHON" -m pip install \
    'numpy==2.2.6' 'opencv-python==5.0.0.93' 'imageio==2.38.0' \
    'imageio-ffmpeg==0.6.0' 'safetensors==0.8.0' 'huggingface_hub==2.1.1' 'PyYAML==6.0.3'

# Install only tracked source from the selected commit, including no weights or
# local experiments. A regular wheel stays independent of checkout edits.
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
git -C "$SOURCE_DIR" archive "$SOURCE_REVISION" | tar -x -C "$BUILD_DIR"
"$WORKER_PYTHON" -m pip install --no-deps --no-build-isolation "$BUILD_DIR"
"$WORKER_PYTHON" -m pip check
"$WORKER_PYTHON" - "$SOURCE_REVISION" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import torch
import mose3
import yaml
from mose3.models.mose3 import MoSE3

assert torch.__version__ == '2.6.0+cu126'
import inspect
assert Path(inspect.getfile(MoSE3)).resolve().is_relative_to(Path(sys.prefix).resolve())
record = {
    'python': sys.executable,
    'source_revision': sys.argv[1],
    'install_mode': 'wheel_from_pinned_git_archive',
    'packages': {name: metadata.version(name) for name in (
        'torch', 'numpy', 'opencv-python', 'imageio', 'imageio-ffmpeg',
        'safetensors', 'huggingface_hub', 'PyYAML', 'mose3')},
    'validation': 'CPU imports passed; weights and GPU inference are separate',
}
(Path(sys.prefix) / 'runtime_setup.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
echo "MoSE3 worker ready: $WORKER_PYTHON"
echo "For a custom location, set GRAIL_MOSE3_PYTHON=$WORKER_PYTHON when running GRAIL."
