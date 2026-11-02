#!/usr/bin/env bash
# Install the pinned public SOMA Retargeter and validate the G1+DEx3 model.
# Usage: bash scripts/setup/install_soma_retargeter.sh [--source DIRECTORY] [--python PYTHON]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/imports/soma-retargeter"
GMR_DIR="$PROJECT_ROOT/imports/GMR"
BASE_PYTHON="${GRAIL_BASE_PYTHON:-python}"
SOURCE_REVISION=cbdb2b494a2af82137bf9b6f15122d60167281c4
GMR_REVISION=bb1bbe40774794fceb2a7c579a3464a28e68c844
DEX3_MODEL_SHA256=167280f50c138ac548c03c85f609ddd96e0d994e9f288d6baa42b67617524ec5

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source|--python)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --source) SOURCE_DIR="$2" ;;
                --python) BASE_PYTHON="$2" ;;
            esac
            shift 2 ;;
        --help|-h) sed -n '2,3s/^# //p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -f "$SOURCE_DIR/pyproject.toml" ] || {
    echo "SOMA Retargeter source missing. Run git submodule update --init imports/soma-retargeter." >&2
    exit 1
}
[ -f "$GMR_DIR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml" ] || {
    echo "GMR G1+DEx3 assets are missing. Run git submodule update --init imports/GMR." >&2
    exit 1
}
GIT_SOURCE=(git -c "safe.directory=$SOURCE_DIR" -C "$SOURCE_DIR")
[ "$("${GIT_SOURCE[@]}" rev-parse HEAD)" = "$SOURCE_REVISION" ] || {
    echo "Expected SOMA Retargeter at $SOURCE_REVISION; update the submodule first." >&2
    exit 1
}
GIT_GMR=(git -c "safe.directory=$GMR_DIR" -C "$GMR_DIR")
[ "$("${GIT_GMR[@]}" rev-parse HEAD)" = "$GMR_REVISION" ] || {
    echo "Expected GMR at $GMR_REVISION; update the submodule first." >&2
    exit 1
}
[ "$(sha256sum "$GMR_DIR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml" | cut -d' ' -f1)" = "$DEX3_MODEL_SHA256" ] || {
    echo "GMR G1+DEx3 model hash differs from the released model." >&2
    exit 1
}
[ -z "$("${GIT_SOURCE[@]}" status --porcelain --untracked-files=no)" ] || {
    echo "SOMA Retargeter has tracked edits; preserving it without installation." >&2
    exit 1
}
command -v git-lfs >/dev/null 2>&1 || {
    echo "git-lfs is required to hydrate the SOMA reference and Unitree G1 assets." >&2
    exit 1
}

"${GIT_SOURCE[@]}" lfs pull \
    --include='soma_retargeter/assets/soma/**,soma_retargeter/assets/robotics/unitree/g1/**'

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    "$BASE_PYTHON" -m pip install 'uv>=0.9,<1'
    UV_BIN="$("$BASE_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.executable).with_name("uv"))')"
fi
"$UV_BIN" sync --frozen --project "$SOURCE_DIR"
"$SOURCE_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

import newton
import warp
from pxr import Usd
from soma_retargeter.io.bvh import BVHImporter
from soma_retargeter.robotics.robot_registry import list_available_soma_targets

assert sys.version_info >= (3, 12)
assert BVHImporter is not None
assert 'unitree_g1' in list_available_soma_targets()
assert newton is not None and warp is not None and Usd is not None
print(
    'SOMA Retargeter Unitree G1 runtime OK: '
    f"python={sys.version_info.major}.{sys.version_info.minor}, "
    f"soma-retargeter={metadata.version('soma-retargeter')}, "
    f"newton={metadata.version('newton')}, warp={metadata.version('warp-lang')}"
)
PY
"$SOURCE_DIR/.venv/bin/python" "$SOURCE_DIR/app/bvh_to_csv_converter.py" --help >/dev/null
"$SOURCE_DIR/.venv/bin/python" "$PROJECT_ROOT/grail/retargeting/soma_target.py" --help >/dev/null
echo "GRAIL G1+DEx3 source-bound target generator OK"
