"""Isolated native geometry worker; invoke in the selected model runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .native import evaluate_motion, load_native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="Canonical native NPZ")
    parser.add_argument("--model-config", required=True, help="JSON configuration file")
    parser.add_argument("--frames", required=True, help="JSON source-frame list or path to JSON")
    parser.add_argument("--output", required=True, help="New geometry NPZ")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.model_config).read_text())
    if Path(args.motion).suffix != ".npz":
        parser.error("Worker motion input must be a safe canonical NPZ")
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("Worker output must end in .npz")
    if output.exists():
        raise FileExistsError(output)
    frame_spec = args.frames.strip()
    frames = json.loads(frame_spec if frame_spec.startswith("[") else Path(frame_spec).read_text())
    motion = load_native(args.motion, config["model_type"])
    result = evaluate_motion(motion, config, frames=frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print(
        json.dumps(
            dict(
                output=str(output.resolve()),
                frames=len(result["frames"]),
                vertices=int(result["vertices"].shape[1]),
            )
        )
    )


if __name__ == "__main__":
    main()
