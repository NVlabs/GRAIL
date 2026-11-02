"""MoGe-3 metric depth adapter with an isolated inference runtime."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from grail.core.depth_config import MOGE_MODEL_ID, MOGE_MODEL_REVISION


def _runtime_python():
    """Keep MoGe's Triton dependencies out of the reconstruction interpreter."""
    root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("GRAIL_MOGE_PYTHON")
    runtime = Path(configured).expanduser() if configured else root / ".venvs/moge3/bin/python"
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise RuntimeError(
            f"MoGe-3 runtime not found: {runtime}. "
            "Run scripts/setup/install_moge3.sh with the GRAIL Python environment, "
            "or set GRAIL_MOGE_PYTHON to an installed MoGe-3 runtime interpreter."
        )
    return runtime


def est_depth(
    image_list,
    device="cuda",
    intrinsics=None,
    gt_depth_first_frame=None,
    return_metadata=False,
):
    """Estimate native-resolution metric camera-Z depth with pinned MoGe-3 ViT-G.

    The foreground subprocess keeps MoGe dependencies outside the reconstruction
    interpreter. Invalid or masked pixels remain zero. GT alignment is performed
    separately by preprocessing; gt_depth_first_frame is kept for compatibility
    and is not supplied to the network. return_metadata adds worker provenance.
    """
    images = [str(Path(p).expanduser().resolve()) for p in image_list]
    if not images:
        raise ValueError("MoGe-3 requires at least one input image")
    for path in images:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if isinstance(intrinsics, (str, os.PathLike)):
        K = np.loadtxt(intrinsics)
    elif intrinsics is not None:
        K = np.asarray(intrinsics)
    else:
        K = None
    if K is not None:
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("intrinsics must be a finite 3x3 matrix with positive focal lengths")
    worker = Path(__file__).with_name("moge_depth_worker.py")
    runtime = _runtime_python()
    request = {
        "images": images,
        "device": str(device),
        "intrinsics": K.tolist() if K is not None else None,
    }
    with tempfile.TemporaryDirectory(prefix="grail-moge3-") as tmp:
        request_path, output_path = Path(tmp) / "request.json", Path(tmp) / "depth.pt"
        request_path.write_text(json.dumps(request))
        subprocess.run(
            [
                str(runtime),
                str(worker),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        result = torch.load(output_path, map_location=device, weights_only=True)
    metadata, depth_list = result["metadata"], result["depth_list"]
    if (
        metadata.get("model_id") != MOGE_MODEL_ID
        or metadata.get("model_revision") != MOGE_MODEL_REVISION
    ):
        raise RuntimeError("MoGe worker returned unexpected model provenance")
    if len(depth_list) != len(images):
        raise RuntimeError("MoGe worker returned a different frame count")
    for i, depth in enumerate(depth_list):
        if depth.ndim != 2 or not torch.isfinite(depth).all() or torch.any(depth < 0):
            raise RuntimeError(f"Invalid MoGe-3 depth tensor for frame {i}")
    return (depth_list, metadata) if return_metadata else depth_list
