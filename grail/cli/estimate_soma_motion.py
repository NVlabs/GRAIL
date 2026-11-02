#!/usr/bin/env python3
"""Estimate source-faithful SOMA03 motion with public GEM-X.

This entrypoint runs human estimation only. It writes native source identity;
the configured G1 identity is applied later by reconstruction placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/recon_4dhoi/manip_soma.yaml")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Persistent content-checked estimator cache; keep between runs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="GRAIL native motion .npz path")
    parser.add_argument("--report", type=Path, help="JSON report path; defaults to OUTPUT.json")
    parser.add_argument("--backend", choices=("gem_x",), help="Select the public GEM-X estimator")
    parser.add_argument(
        "--human-crop-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SAM2-assisted crops (default on); disable explicitly for a control",
    )
    parser.add_argument(
        "--human-crop-masks",
        type=Path,
        help="Source-aligned SAM2 masks required by default crop repair",
    )
    parser.add_argument(
        "--human-crop-receipt",
        type=Path,
        help="SAM2 completion.json; defaults to the mask file's sibling",
    )
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    video, config = args.video.expanduser().resolve(strict=True), args.config.expanduser().resolve(
        strict=True
    )
    output, cache = args.output.expanduser().resolve(), args.cache_dir.expanduser().resolve()
    report_path = (args.report or output.with_suffix(".json")).expanduser().resolve()
    if report_path in {video, config, output}:
        parser.error("--report must be distinct from the source, config, and NPZ output")

    # Keep --help cheap and avoid importing the HOI optimizer and object tracker.
    import numpy as np

    from grail.core.config import load_recon_config
    from grail.core.io import save_human_motion_data
    from grail.core.torch_utils import tensor_to_numpy
    from grail.pose_est.human_pose import run_human_pose_est

    cfg, _ = load_recon_config(str(config))
    model_cfg = dict(cfg["human_model"])
    if model_cfg.get("body_model") != "soma03":
        parser.error("--config must explicitly select human_model.body_model: soma03")
    if args.backend:
        model_cfg["human_pose_backend"] = args.backend
    backend = model_cfg.get("human_pose_backend", "gem_x")
    if backend != "gem_x":
        parser.error("human_pose_backend must be gem_x; the legacy predictor has been removed")
    if args.human_crop_repair is not None:
        model_cfg[f"{backend}_human_crop_repair"] = args.human_crop_repair
    if args.human_crop_masks:
        model_cfg[f"{backend}_human_crop_masks"] = str(args.human_crop_masks.resolve(strict=True))
    if args.human_crop_receipt:
        model_cfg[f"{backend}_human_crop_receipt"] = str(
            args.human_crop_receipt.resolve(strict=True)
        )
    if model_cfg.get(f"{backend}_human_crop_repair", True) and not model_cfg.get(
        f"{backend}_human_crop_masks"
    ):
        parser.error(
            "Default crop repair requires --human-crop-masks and its receipt; "
            "use --no-human-crop-repair only for an explicit detector-only control"
        )
    motion = run_human_pose_est(
        str(video), model="soma03", cache_dir=str(cache), human_model_config=model_cfg
    )
    converted = {
        space: tensor_to_numpy(motion[space]) for space in ("motion_global", "motion_incam")
    }
    incam = converted["motion_incam"]
    diagnostics = {}
    for space, data in converted.items():
        fields = (
            "poses",
            "trans",
            "identity_coeffs",
            "scale_params",
            "left_hand_pose",
            "right_hand_pose",
        )
        finite = all(np.isfinite(data[key]).all() for key in fields)
        if not finite:
            raise ValueError(f"Nonfinite estimated parameters in {space}")
        diagnostics[space] = {
            "finite": True,
            "shapes": {key: list(data[key].shape) for key in fields},
            "global_scale_exact_one": bool(np.all(data["scale_params"][..., 0] == 1)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.npz")
    save_human_motion_data(converted["motion_global"], incam, str(temporary))
    temporary.replace(output)
    report = {
        "schema": "grail.soma03.estimation_report.v1",
        "status": "completed",
        "source_video": str(video),
        "source_sha256": sha256(video),
        "config": str(config),
        "config_sha256": sha256(config),
        "output": str(output),
        "output_sha256": sha256(output),
        "human_pose_backend": backend,
        "model_id": "gem_x_native77",
        "frame_count": int(incam["poses"].shape[0]),
        "fps": float(incam["mocap_frame_rate"]),
        "predicted_body_height_m": float(incam["predicted_body_height"]),
        "fixed_identity_applied": False,
        "configured_downstream_identity": model_cfg.get("soma_identity_params_path"),
        "diagnostics": diagnostics,
        "estimator_provenance": incam["estimator_provenance"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary_report.replace(report_path)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "report": str(report_path),
                "frame_count": report["frame_count"],
                "fps": report["fps"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
