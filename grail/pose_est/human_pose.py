"""Human pose estimation with the native GEM-X SOMA03 predictor."""

import numpy as np


def run_human_pose_est_soma03(video_path, cache_dir, human_model_config):
    """Estimate native SOMA03 motion before any fixed G1 identity substitution."""
    from grail.models.soma03_model import Soma03HumanModel

    cfg = dict(human_model_config or {})
    backend = cfg.get("human_pose_backend", "gem_x")
    if backend != "gem_x":
        raise ValueError(
            f"Unsupported SOMA03 human_pose_backend: {backend!r}; use the public GEM-X backend"
        )
    from grail.adapters.gem_x_soma import infer_human_pose, prediction_to_motion

    for field in ("gem_x_source_root", "gem_x_config_path", "gem_x_python"):
        if not cfg.get(field):
            raise ValueError(f"human_model.{field} is required for human_pose_backend=gem_x")
    prediction = infer_human_pose(
        video_path,
        cache_dir,
        source_root=cfg["gem_x_source_root"],
        config_path=cfg["gem_x_config_path"],
        python=cfg["gem_x_python"],
        device=cfg.get("gem_x_device", "cuda"),
        is_static_cam=cfg.get("static_cam", True),
        human_crop_repair=cfg.get("gem_x_human_crop_repair", True),
        human_crop_masks=cfg.get("gem_x_human_crop_masks"),
        human_crop_receipt=cfg.get("gem_x_human_crop_receipt"),
        human_crop_config=cfg.get("gem_x_human_crop_config"),
        postprocess=cfg.get("gem_x_postprocess", False),
        soma_assets=cfg.get("gem_x_soma_assets"),
    )
    motion = prediction_to_motion(prediction)

    # Source height is derived from the predicted identity, never the fixed G1
    # identity. Downstream placement consumes this stored height when swapping
    # identities; it must not mutate the native fixed-global-scale channel.
    source_cfg = dict(cfg)
    source_cfg.pop("soma_identity_params_path", None)
    source_model = Soma03HumanModel(source_cfg, device="cpu")
    source_height = float(source_model.compute_tpose_height(motion["motion_incam"]))
    if not np.isfinite(source_height) or source_height <= 0:
        raise ValueError("Predicted SOMA03 source identity has an invalid body height")
    for space in ("motion_incam", "motion_global"):
        motion[space]["predicted_body_height"] = source_height
        motion[space]["estimator_provenance"]["predicted_body_height_m"] = source_height
        motion[space]["estimator_provenance"]["predicted_height_identity"] = "native_estimator"
    return motion


def run_human_pose_est(
    video_path,
    model="soma03",
    cache_dir="cache",
    *,
    human_model_config=None,
):
    """Predict with the selected native estimator; never convert/fall back silently."""
    if model != "soma03":
        raise ValueError(f"Unsupported body model {model!r}; prediction requires soma03")
    if human_model_config is None:
        from grail.core.config import load_recon_config

        cfg, _ = load_recon_config(None)
        human_model_config = cfg["human_model"]
    if human_model_config.get("body_model", "soma03") != model:
        raise ValueError("Human model configuration and requested backend must agree")
    return run_human_pose_est_soma03(
        video_path, cache_dir=cache_dir, human_model_config=human_model_config
    )


if __name__ == "__main__":
    from grail.cli.estimate_soma_motion import main

    raise SystemExit(main())
