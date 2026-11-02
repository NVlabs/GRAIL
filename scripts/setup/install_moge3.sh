#!/usr/bin/env bash
# Install the inference-only MoGe-3 worker without replacing GRAIL's PyTorch or
# Triton. Activate the grail environment first, or pass --python /path/to/python.
# Usage: bash scripts/setup/install_moge3.sh [--python PYTHON] [--venv DIRECTORY] [--source DIRECTORY] [--no-gpu-import]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${GRAIL_BASE_PYTHON:-python}"
VENV_DIR="$PROJECT_ROOT/.venvs/moge3"
MOGE_DIR="$PROJECT_ROOT/imports/MoGe"
MOGE_REVISION=74fbce054ebed49800de42d0ad0e83495065719a
UTILS_REVISION=62f09d58509485564e24d5d9f6aac9ee9ebc0c37
FLEX_REVISION=b2fadb29d41846c7981ade6801ffc689fae119cf
IMPORT_CHECK_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python|--venv|--source)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --python) BASE_PYTHON="$2" ;;
                --venv) VENV_DIR="$2" ;;
                --source) MOGE_DIR="$2" ;;
            esac
            shift 2 ;;
        --no-gpu-import) IMPORT_CHECK_ARGS=(--no-gpu-import); shift ;;
        --help|-h) sed -n '2,4s/^# //p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
BASE_PYTHON="$("$BASE_PYTHON" -c 'import sys; print(sys.executable)')"
VENV_DIR="$("$BASE_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).absolute())' "$VENV_DIR")"
[ -f "$MOGE_DIR/moge/model/v3.py" ] || {
    echo "MoGe-3 source missing. Run git submodule update --init imports/MoGe." >&2
    exit 1
}
[ "$(git -C "$MOGE_DIR" rev-parse HEAD)" = "$MOGE_REVISION" ] || {
    echo "Expected imports/MoGe at $MOGE_REVISION; update the submodule before installing." >&2
    exit 1
}
[ -z "$(git -C "$MOGE_DIR" status --porcelain --untracked-files=no)" ] || {
    echo "MoGe tracked source has local edits; preserve it and use the pinned submodule." >&2
    exit 1
}

# Verify the shared environment before touching the worker. Installing with
# --no-deps below deliberately prevents pip from upgrading its torch/CUDA stack.
BASE_VERSIONS="$("$BASE_PYTHON" - <<'PY'
import json, sys, torch, torchvision, triton, cv2, numpy, scipy, PIL, huggingface_hub
if sys.version_info < (3, 10):
    raise SystemExit('MoGe-3 requires Python >= 3.10.')
if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 4):
    raise SystemExit('MoGe-3 requires torch >= 2.4 in the base environment.')
print(json.dumps({'torch': torch.__version__, 'triton': triton.__version__}, sort_keys=True))
PY
)"

if [ -e "$VENV_DIR" ]; then
    [ -f "$VENV_DIR/pyvenv.cfg" ] && [ -x "$VENV_DIR/bin/python" ] || {
        echo "Refusing to replace existing non-venv path: $VENV_DIR" >&2; exit 1;
    }
    "$VENV_DIR/bin/python" - "$BASE_PYTHON" "$VENV_DIR" "$BASE_VERSIONS" "$MOGE_REVISION" "$UTILS_REVISION" "$FLEX_REVISION" <<'PY'
from pathlib import Path
import json
import sys
if Path(sys._base_executable).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Existing MoGe worker uses a different base Python; select a new --venv directory.')
cfg = (Path(sys.argv[2]) / 'pyvenv.cfg').read_text().lower()
if 'include-system-site-packages = true' not in cfg:
    raise SystemExit('Existing MoGe worker must be created with --system-site-packages.')
identity = {
    'base_python': str(Path(sys.argv[1]).resolve()),
    'base_versions': json.loads(sys.argv[3]),
    'moge_revision': sys.argv[4],
    'utils3d_moge_revision': sys.argv[5],
    'flex_gemm_revision': sys.argv[6],
}
marker = Path(sys.argv[2]) / '.grail-moge3-setup.json'
legacy = Path(sys.argv[2]) / 'runtime_setup.json'
if marker.is_file():
    if json.loads(marker.read_text()) != identity:
        raise SystemExit('Preserving differently configured MoGe environment; select a new --venv.')
elif legacy.is_file():
    receipt = json.loads(legacy.read_text())
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise SystemExit('Preserving MoGe environment with a mismatched legacy receipt.')
else:
    raise SystemExit('Refusing to modify an unowned MoGe environment; select a new --venv.')
PY
else
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

# Mark ownership before the first pip install so an interrupted owned setup can
# be resumed, while a caller's unrelated shared environment remains untouched.
"$BASE_PYTHON" - "$VENV_DIR" "$BASE_PYTHON" "$BASE_VERSIONS" "$MOGE_REVISION" "$UTILS_REVISION" "$FLEX_REVISION" <<'PY'
import json
from pathlib import Path
import sys
identity = {
    'base_python': str(Path(sys.argv[2]).resolve()),
    'base_versions': json.loads(sys.argv[3]),
    'moge_revision': sys.argv[4],
    'utils3d_moge_revision': sys.argv[5],
    'flex_gemm_revision': sys.argv[6],
}
(Path(sys.argv[1]) / '.grail-moge3-setup.json').write_text(json.dumps(identity, indent=2) + '\n')
PY

WORKER_PYTHON="$VENV_DIR/bin/python"
export PIP_REQUIRE_VIRTUALENV=true
"$WORKER_PYTHON" -m pip install --no-deps 'setuptools==79.0.1' 'triton==3.4.0'

# Skip already-pinned Git dependencies on repeat invocations. These packages are
# inference dependencies only; MoGe's training and Gradio extras are not needed.
install_git_dependency() {
    local package="$1" url="$2" revision="$3"
    if ! "$WORKER_PYTHON" - "$package" "$revision" <<'PY'
import importlib.metadata as metadata
import json, sys
try:
    direct = json.loads(metadata.distribution(sys.argv[1]).read_text('direct_url.json') or '{}')
except metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if direct.get('vcs_info', {}).get('commit_id') == sys.argv[2] else 1)
PY
    then
        "$WORKER_PYTHON" -m pip install --no-deps --no-build-isolation "git+$url@$revision"
    fi
}
install_git_dependency utils3d_moge https://github.com/EasternJournalist/utils3d-moge.git "$UTILS_REVISION"
install_git_dependency flex_gemm https://github.com/JeffreyXiang/FlexGEMM.git "$FLEX_REVISION"
"$WORKER_PYTHON" -m pip install --no-deps --no-build-isolation -e "$MOGE_DIR"

# FlexGEMM queries the CUDA device during import. Image construction explicitly
# defers those imports; it still validates pinned distributions and source code.
# The default install and deployment setup use real GPU-dependent import checks.
"$WORKER_PYTHON" "$PROJECT_ROOT/scripts/setup/check_moge3_runtime.py" \
    --source "$MOGE_DIR" --base-versions "$BASE_VERSIONS" \
    --output "$VENV_DIR/runtime_setup.json" "${IMPORT_CHECK_ARGS[@]}"
AFTER_VERSIONS="$("$BASE_PYTHON" -c 'import json,torch,triton; print(json.dumps({"torch":torch.__version__,"triton":triton.__version__},sort_keys=True))')"
[ "$BASE_VERSIONS" = "$AFTER_VERSIONS" ] || {
    echo "Base torch/Triton versions unexpectedly changed." >&2; exit 1;
}
echo "MoGe-3 worker installed: $WORKER_PYTHON (validation details in runtime_setup.json)"
echo "For a custom location, set GRAIL_MOGE_PYTHON=$WORKER_PYTHON when running GRAIL."
