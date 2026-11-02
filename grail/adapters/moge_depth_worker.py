"""Foreground subprocess entry point for the isolated MoGe-3 runtime."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "imports/MoGe"))

import cv2
from huggingface_hub import hf_hub_download
import torch
from tqdm import tqdm

from grail.core.depth_config import (
    MOGE_MODEL_ID,
    MOGE_MODEL_REVISION,
    MOGE_REFINE_STEPS,
    MOGE_RESOLUTION_LEVEL,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_images(model, images, device, intrinsics):
    """Return CPU tensors so the entire clip never accumulates on the GPU."""
    depths, records = [], []
    first_shape = None
    for path in tqdm(images, desc="MoGe-3 depth"):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read input image: {path}")
        if first_shape is None:
            first_shape = bgr.shape[:2]
        if bgr.shape[:2] != first_shape:
            raise ValueError("Video depth input frames must have the same image dimensions")
        h, w = first_shape
        fov_x = (
            None if intrinsics is None else math.degrees(2 * math.atan(w / (2 * intrinsics[0][0])))
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Match the reviewed MoGe-3 benchmark's float32 device normalization.
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device).float() / 255.0
        is_cuda = torch.device(device).type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.infer(
                image,
                fov_x=fov_x,
                resolution_level=MOGE_RESOLUTION_LEVEL,
                use_fp16=is_cuda,
                refine_steps=MOGE_REFINE_STEPS,
                force_projection=True,
                apply_mask=True,
            )
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        depth = output["depth"].float()
        if tuple(depth.shape) != (h, w):
            raise RuntimeError(
                f"Unexpected MoGe-3 output resolution {depth.shape}; expected {(h, w)}"
            )
        valid = torch.isfinite(depth) & (depth > 0)
        if "mask" in output:
            valid &= output["mask"].bool()
        depth = torch.where(valid, depth, 0).cpu()
        depths.append(depth)
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "shape": [h, w],
                "fov_x_degrees": fov_x,
                "invalid_pixels": int((~valid).sum().item()),
                "inference_seconds": elapsed,
                "peak_gpu_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if is_cuda else 0
                ),
            }
        )
        del image, output, valid
    return depths, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    from moge.model.v3 import MoGeModel

    checkpoint = hf_hub_download(MOGE_MODEL_ID, "model.pt", revision=MOGE_MODEL_REVISION)
    device = request["device"]
    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()
    depths, records = infer_images(model, request["images"], device, request["intrinsics"])
    try:
        code_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "imports/MoGe"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = None
    metadata = {
        "model_id": MOGE_MODEL_ID,
        "model_revision": MOGE_MODEL_REVISION,
        "checkpoint_sha256": _sha256(checkpoint),
        "code_revision": code_revision,
        "worker_sha256": _sha256(__file__),
        "python": sys.executable,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "depth_domain": "metric_camera_z_meters",
        "resolution_level": MOGE_RESOLUTION_LEVEL,
        "refine_steps": MOGE_REFINE_STEPS,
        "use_fp16": torch.device(device).type == "cuda",
        "seed": 20260923,
        "normalization": "RGB uint8 to device float32, then divide by 255",
        "intrinsics": request["intrinsics"],
        "frames": records,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    torch.save({"depth_list": depths, "metadata": metadata}, args.output)
    print(f"MoGe-3: produced {len(depths)} metric depth frames", flush=True)


if __name__ == "__main__":
    main()
