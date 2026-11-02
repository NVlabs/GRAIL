"""
GEM-SMPL human pose estimation adapter (grail branch).

Provides `infer_human_pose()` using the hmr4d package from the GEM-SMPL
grail branch (GENMO architecture with SMPL-X body model).

Reuses `demo_slam.py` from GEM-SMPL for preprocessing and data loading.
"""

import os
import shutil
import sys
import time
import types
from pathlib import Path

import torch

from grail.core.video import normalize_motion_fps

_GEM_SMPL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "imports", "GEM-SMPL")


def _setup_imports():
    """Add GEM-SMPL demo to sys.path and mock problematic modules."""
    demo_dir = os.path.join(_GEM_SMPL_ROOT, "tools", "demo")
    if os.path.abspath(_GEM_SMPL_ROOT) not in sys.path:
        sys.path.insert(0, os.path.abspath(_GEM_SMPL_ROOT))
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

    # Mock modules that demo_slam.py imports at module level but we don't need
    # (visualization, rendering, remote sync utilities)
    if "motiondiff.utils.vis_scenepic" not in sys.modules:
        mock = types.ModuleType("motiondiff.utils.vis_scenepic")
        mock.ScenepicVisualizer = lambda *a, **kw: None
        sys.modules["motiondiff.utils.vis_scenepic"] = mock

    if "motiondiff.utils.tools" not in sys.modules:

        class _CatchAllMock(types.ModuleType):
            """Returns a no-op for any attribute access (Timer, wandb_run_exists, etc.)."""

            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return lambda *a, **kw: None

        sys.modules["motiondiff.utils.tools"] = _CatchAllMock("motiondiff.utils.tools")

    if "hmr4d.utils.vis.o3d_render" not in sys.modules:
        mock = types.ModuleType("hmr4d.utils.vis.o3d_render")
        mock.Settings = type("Settings", (), {"Transparency": 0, "LIT": 1})
        mock.create_meshes = lambda *a, **kw: None
        mock.get_ground = lambda *a, **kw: None
        sys.modules["hmr4d.utils.vis.o3d_render"] = mock


def infer_human_pose(
    video_path,
    cache_dir,
    is_static_cam=False,
    verbose=False,
    *,
    human_crop_repair=False,
    human_crop_masks=None,
    human_crop_receipt=None,
    human_crop_config=None,
):
    """
    Run GEM-SMPL (hmr4d) human pose estimation on a video.

    Returns a dict compatible with run_human_pose_est.py:
        {
            "smpl_params_global": {body_pose, global_orient, transl, betas},
            "smpl_params_incam":  {body_pose, global_orient, transl, betas},
            "vitpose": (L, 17, 3),
            "foot_contact_probs": (L, 4) or None,
        }
    """
    # Low-level callers retain the detector-only control. Pipeline YAML explicitly
    # enables repair, with a completed source-bound SAM2 prediction as input.
    from grail.adapters.gem_smpl_crop import (
        completed_cache,
        file_binding,
        prediction_request,
        prepare_crop_repair,
        prepare_run_directory,
        preprocess_cache_paths,
        read_video_timing,
        record_preprocess_cache,
        request_key,
        seed_preprocess_crops,
        validate_preprocess_cache,
        write_json,
    )

    if type(human_crop_repair) is not bool:
        raise ValueError("human_crop_repair must be a boolean")
    if human_crop_repair and not human_crop_masks:
        raise ValueError("GEM-SMPL crop repair requires source-bound human_crop_masks")
    t0 = time.time()
    video_path_abs = os.path.abspath(video_path)
    timing = read_video_timing(video_path_abs)
    crop_request = (
        prepare_crop_repair(
            video_path_abs,
            human_crop_masks,
            human_crop_receipt,
            timing=timing,
            config=human_crop_config,
        )
        if human_crop_repair
        else None
    )

    # Setup imports and run directly
    _setup_imports()

    import cv2
    import hmr4d.model.genmo.genmo_demo  # noqa: F401 — registers genmo_demo model config
    from hmr4d import PROJ_ROOT

    if Path(PROJ_ROOT).resolve() != Path(_GEM_SMPL_ROOT).resolve():
        raise ValueError("Loaded hmr4d package does not belong to the selected GEM-SMPL checkout")
    import hydra
    from demo_slam import Tracker, get_bbx_xys_from_xyxy, load_data_dict, run_preprocess
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    # Build Hydra config (replicates parse_args_to_cfg without argparse)
    video_path_obj = Path(video_path_abs)
    assert video_path_obj.exists(), f"Video not found at {video_path_obj}"
    length, width, height = get_video_lwh(video_path_obj)
    capture = cv2.VideoCapture(str(video_path_obj))
    orig_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    orig_fps = normalize_motion_fps(orig_fps)
    if (length, width, height) != (timing["frame_count"], timing["width"], timing["height"]) or abs(
        orig_fps - timing["fps"]
    ) > 1e-3:
        raise ValueError("GEM-SMPL decoder and bound source timeline differ")
    Log.info(f"[GEM-SMPL] Input: {video_path_obj}, (L, W, H) = ({length}, {width}, {height})")

    register_store_gvhmr()
    overrides = [
        f"video_name={video_path_obj.stem}",
        f"static_cam={is_static_cam}",
        f"verbose={verbose}",
        f"output_root={os.path.abspath(cache_dir)}",
    ]
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo_genmo", overrides=overrides)

    # Override ckpt_path to absolute path so it doesn't depend on CWD
    gem_smpl_root = os.path.abspath(_GEM_SMPL_ROOT)
    cfg.ckpt_path = os.path.join(gem_smpl_root, cfg.ckpt_path)
    model_config = OmegaConf.to_container(cfg, resolve=True)
    # Output paths are addresses, not model settings. The content-keyed directory
    # below owns every vendor feature and HMR cache, including detector-only runs.
    for name in ("output_root", "output_dir", "preprocess_dir", "video_path", "paths", "verbose"):
        model_config.pop(name, None)
    request = prediction_request(
        video_path_abs,
        gem_smpl_root,
        cfg.ckpt_path,
        model_config,
        source_timing=timing,
        crop_request=crop_request,
        static_cam=is_static_cam,
    )
    run_dir = prepare_run_directory(cache_dir, request)
    cached_results = run_dir / "gem_smpl_pred.pt"
    if completed_cache(run_dir, request):
        print(f"[GEM-SMPL] Loading bound cached results from {cached_results}")
        return torch.load(cached_results, map_location="cpu", weights_only=False)
    cfg.output_dir = str(run_dir)
    cfg.preprocess_dir = str(run_dir / "preprocess")
    cfg.video_path = str(run_dir / "0_input_video.mp4")

    paths = cfg.paths
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Preserve source pixels/timeline; never reuse an old same-length video.
    if not Path(cfg.video_path).exists():
        shutil.copyfile(video_path_obj, cfg.video_path)
    if file_binding(cfg.video_path)["sha256"] != request["source"]["sha256"]:
        raise ValueError("Cached GEM-SMPL video differs from its bound original source")

    # GEM-SMPL code uses bare "inputs/checkpoints/..." paths from CWD,
    # so temporarily chdir to GEM-SMPL root while running its functions.
    prev_cwd = os.getcwd()
    os.chdir(gem_smpl_root)
    try:
        # Seed effective crops before any crop-dependent vendor feature/model.
        # bbx_original.pt retains the detector output for diagnostics and controls.
        validate_preprocess_cache(cfg, request)
        seed_preprocess_crops(cfg, request, Tracker, get_bbx_xys_from_xyxy)
        run_preprocess(cfg, orig_fps)
        record_preprocess_cache(cfg, request)

        # Load preprocessed data
        data = load_data_dict(cfg)

        # Run HMR4D inference
        if not Path(paths.hmr4d_results).exists():
            Log.info("[GEM-SMPL] Running HMR4D prediction")
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            torch.save(pred, paths.hmr4d_results)
            write_json(
                run_dir / "hmr_completion.json",
                {
                    "request_sha256": request_key(request),
                    "output": file_binding(paths.hmr4d_results),
                },
            )
            Log.info(f"[GEM-SMPL] Saved HMR4D results to {paths.hmr4d_results}")
        else:
            import json

            receipt = run_dir / "hmr_completion.json"
            expected = {
                "request_sha256": request_key(request),
                "output": file_binding(paths.hmr4d_results),
            }
            if not receipt.exists() or json.loads(receipt.read_text()) != expected:
                raise ValueError("Unbound or modified GEM-SMPL HMR result")
            Log.info(f"[GEM-SMPL] Loading cached HMR4D results from {paths.hmr4d_results}")
            pred = torch.load(paths.hmr4d_results, map_location="cpu")
    finally:
        os.chdir(prev_cwd)

    # Load vitpose
    vitpose = None
    if os.path.exists(paths.vitpose):
        vitpose = torch.load(paths.vitpose, map_location="cpu")
        if isinstance(vitpose, tuple):
            vitpose = vitpose[0]

    # Remap to expected output format
    result = {
        "smpl_params_global": pred.get("smpl_params_global", {}),
        "smpl_params_incam": pred.get("smpl_params_incam", {}),
        "vitpose": pred.get("vitpose", vitpose),
        "foot_contact_probs": pred.get("foot_contact_probs", None),
        "estimator_provenance": {
            "model": "gem_smpl",
            "native_body_model": "smplx",
            "request_sha256": request_key(request),
            "request": str(run_dir / "request.json"),
            "human_crop_repair": human_crop_repair,
            "human_crop_repair_report": str(run_dir / "human_crop_repair.json"),
            "source_sha256": request["source"]["sha256"],
        },
    }

    # Extract foot contact probs if not already in pred
    if result["foot_contact_probs"] is None:
        if "net_outputs" in pred and "model_output" in pred.get("net_outputs", {}):
            model_output = pred["net_outputs"]["model_output"]
            if "static_conf_logits" in model_output:
                static_conf_logits = model_output["static_conf_logits"]
                result["foot_contact_probs"] = torch.sigmoid(static_conf_logits[:, :, :4])

    # Ensure CPU tensors
    for key in ["smpl_params_global", "smpl_params_incam"]:
        if isinstance(result[key], dict):
            result[key] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result[key].items()
            }

    # Cache results
    torch.save(result, cached_results)
    outputs = {
        "prediction": file_binding(cached_results),
        "input_video": file_binding(cfg.video_path),
        "original_bbox": file_binding(run_dir / "preprocess/bbx_original.pt"),
        "crop_report": file_binding(run_dir / "human_crop_repair.json"),
    }
    for name, path in preprocess_cache_paths(cfg, request).items():
        outputs[name] = file_binding(path)
    outputs["hmr4d_results"] = file_binding(paths.hmr4d_results)
    if file_binding(video_path_abs) != request["source"]:
        raise ValueError("GEM-SMPL source changed during inference")
    if (
        crop_request is not None
        and prepare_crop_repair(
            video_path_abs,
            human_crop_masks,
            human_crop_receipt,
            timing=timing,
            config=human_crop_config,
        )
        != crop_request
    ):
        raise ValueError("GEM-SMPL crop inputs changed during inference")
    write_json(
        run_dir / "completion.json",
        {
            "status": "PREDICTION_COMPLETE",
            "request_sha256": request_key(request),
            "outputs": outputs,
            "elapsed_seconds": time.time() - t0,
        },
    )
    print(f"[GEM-SMPL] Results cached to {cached_results}")
    print(f"[GEM-SMPL] Total time: {time.time() - t0:.1f}s")

    return result
