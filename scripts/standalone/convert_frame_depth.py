"""Convert a single Blender Z-pass (metres) to a registered uint16 PNG (millimetres)."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def convert(source, target):
    if target.exists():
        raise FileExistsError(f"Depth PNG already exists: {target}")
    depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if depth is None or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"Expected a floating-point Blender Z-pass at {source}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    metres = np.clip(np.nan_to_num(depth, nan=0.0, posinf=65.535), 0, 65.535)
    mm = np.rint(metres * 1000).astype("uint16")
    if not cv2.imwrite(str(target), mm):
        raise OSError(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    convert(args.source, args.target)
