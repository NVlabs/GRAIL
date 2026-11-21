"""Isolated optional GraspGenX worker; always emits unreviewed proposals."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

if not __package__:
    folder = Path(__file__).resolve().parent
    package_name = "_grail_graspgen_worker_" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    __package__ = package_name

from . import graspgen_runtime as runtime_tools


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _positive_int(request, name, default, maximum):
    value = request.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _load_backend(runtime, gripper_dir):
    """Import only the verified package; bypass unrelated modules in its checkout."""
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = runtime["checkpoint_dir"]
    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(gripper_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise ValueError("GraspGenX requires a local CUDA-capable PyTorch runtime")
    index = int(runtime["device"].split(":")[1])
    if index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device is unavailable: " + runtime["device"])
    torch.cuda.set_device(index)
    path = Path(runtime["repo"]) / "graspgenx" / "__init__.py"
    spec = importlib.util.spec_from_file_location("graspgenx", path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["graspgenx"] = module
    spec.loader.exec_module(module)
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.grasp_server import GraspGenXSampler
    # Import the serving package too: from_sweep_volume reaches its wire types
    # lazily, and its msgpack/ZMQ dependencies must fail during preflight.
    from graspgenx.serving.types import SweepVolumeParams  # noqa: F401
    return np, torch, load_model_cfg, GraspGenXSampler


def _versions():
    result = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("numpy", "torch", "diffusers", "omegaconf", "huggingface-hub", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _validate_descriptor(descriptor, np):
    if not isinstance(descriptor, dict):
        raise ValueError("Hand descriptor must be a JSON object")
    for key in ("extents_open", "offset_open", "extents_mid", "offset_mid"):
        value = np.asarray(descriptor.get(key), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Invalid hand descriptor vector: " + key)
        if key.startswith("extents") and np.any(value <= 0):
            raise ValueError("Hand descriptor extents must be positive")
    if "gripper_type" in descriptor and (isinstance(descriptor["gripper_type"], bool)
            or not isinstance(descriptor["gripper_type"], int) or descriptor["gripper_type"] not in (0, 1, 2)):
        raise ValueError("gripper_type must be 0, 1 or 2")
    if "fingertip_depth" in descriptor:
        value = float(descriptor["fingertip_depth"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("fingertip_depth must be finite and positive")


def validate_proposals(grasps, scores, count, np):
    """Reject incomplete, nonfinite or nonrigid model output before export."""
    g, s = np.asarray(grasps), np.asarray(scores).reshape(-1)
    if g.shape != (count, 4, 4) or s.shape != (count,):
        raise ValueError("GraspGenX returned an unexpected proposal count or shape")
    if not np.isfinite(g).all() or not np.isfinite(s).all():
        raise ValueError("GraspGenX returned nonfinite proposals or confidence")
    if not np.allclose(g[:, 3], [0, 0, 0, 1], atol=2e-5, rtol=0):
        raise ValueError("GraspGenX returned invalid homogeneous transforms")
    rotations = g[:, :3, :3]
    if (not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=2e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=2e-4, rtol=0)):
        raise ValueError("GraspGenX returned nonrigid or reflected rotations")
    return g, s


def _generate(request, runtime, output, backend, report):
    np, torch, load_model_cfg, sampler_type = backend
    count = _positive_int(request, "num_grasps", 2000, 100000)
    batch_size = _positive_int(request, "batch_size", 200, 2000)
    seed = request.get("seed", 20260929)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    input_path = Path(request["input"]).resolve(strict=True)
    descriptor_path = Path(request["descriptor"]).resolve(strict=True)
    input_hash = runtime_tools.file_hash(input_path)
    descriptor_hash = runtime_tools.file_hash(descriptor_path)
    with np.load(input_path, allow_pickle=False) as data:
        pc = np.asarray(data["object_pc"], dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1:] != (3,) or not 32 <= len(pc) <= 131072 or not np.isfinite(pc).all():
        raise ValueError("Prepared object_pc must contain 32–131072 finite XYZ points in physical object-local metres")
    if np.linalg.norm(np.ptp(pc, axis=0)) <= 1e-8:
        raise ValueError("Prepared object point cloud is degenerate")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    _validate_descriptor(descriptor, np)
    report["inputs"] = dict(input=str(input_path), input_sha256=input_hash,
                            descriptor=str(descriptor_path), descriptor_sha256=descriptor_hash,
                            object_pc_sha256=hashlib.sha256(pc.tobytes()).hexdigest(),
                            point_count=len(pc), units="metres", frame="physical_object_local")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    checkpoints = Path(runtime["checkpoint_dir"])
    started = time.perf_counter()
    # Explicit filenames prevent the upstream latest-epoch finder from loading
    # an additional unverified binary supplied in the same checkpoint folder.
    cfg = load_model_cfg(checkpoints / "release/gen", checkpoints / "release/dis",
                         gen_pth="epoch_736.pth", dis_pth="epoch_1056.pth")
    sampler = sampler_type.from_sweep_volume(cfg, descriptor)
    torch.cuda.synchronize()
    report["model_load_seconds"] = time.perf_counter() - started
    report["recipe"] = dict(num_grasps=count, batch_size=batch_size, seed=seed,
                            grasp_threshold=-1, min_grasps=1, max_tries=1, remove_outliers=False,
                            native_ddpm_steps=int(cfg.diffusion.num_diffusion_iters_eval),
                            gripper_backbone=str(cfg.diffusion.gripper_backbone))
    groups, scores, timings = [], [], []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        started = time.perf_counter()
        grasps, confidence = sampler_type.run_inference(
            pc, sampler, grasp_threshold=-1, num_grasps=size, topk_num_grasps=size,
            min_grasps=1, max_tries=1, remove_outliers=False)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        g, s = validate_proposals(grasps.detach().cpu().numpy(), confidence.detach().cpu().numpy(), size, np)
        groups.append(g)
        scores.append(s)
        _json(output / "progress.json", dict(status="GENERATING", selected=False,
                                             completed=start + size, total=count, batch_seconds=timings[-1]))
    g, s = validate_proposals(np.concatenate(groups), np.concatenate(scores), count, np)
    if runtime_tools.file_hash(input_path) != input_hash or runtime_tools.file_hash(descriptor_path) != descriptor_hash:
        raise ValueError("GraspGenX inputs changed during generation")
    revalidation_start = time.perf_counter()
    if runtime_tools.verify_assets(runtime) != report["assets"]:
        raise ValueError("Pinned GraspGenX assets changed during generation")
    report["asset_revalidation_seconds"] = time.perf_counter() - revalidation_start
    report["checks"]["assets_unchanged"] = True
    with (output / "proposals.npz").open("xb") as stream:
        np.savez_compressed(stream, grasps_object=g, confidence=s)
    report.update(status="PROPOSALS_UNREVIEWED", count=count, selected=False,
                  batch_seconds=timings, generation_seconds=sum(timings),
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  confidence_min=float(s.min()), confidence_max=float(s.max()),
                  proposals=str(output / "proposals.npz"),
                  proposals_sha256=runtime_tools.file_hash(output / "proposals.npz"),
                  physical_contact_validated=False, native_pose_validated=False,
                  sequence_validated=False)


def execute(request):
    """Run one request. Failure receipts never overwrite an existing output."""
    started, output, output_created = time.perf_counter(), None, False
    report = dict(schema_version=1, ok=False, status="FAILED", selected=False, checks={}, errors=[])
    try:
        if not isinstance(request, dict) or request.get("action") not in ("preflight", "run"):
            raise ValueError("action must be preflight or run")
        if request["action"] == "run":
            output = Path(request["output"]).expanduser().resolve()
            for key in ("repo", "checkpoint_dir"):
                supplied = request.get("runtime", {}).get(key)
                if isinstance(supplied, str):
                    protected = Path(supplied).expanduser().resolve()
                    if output == protected or protected in output.parents or output in protected.parents:
                        raise ValueError("Output overlaps the external " + key + ": " + str(protected))
            output.mkdir(parents=True, exist_ok=False)
            output_created = True
        runtime = runtime_tools.validate_runtime(request["runtime"])
        report["runtime"] = runtime
        asset_start = time.perf_counter()
        report["assets"] = runtime_tools.verify_assets(runtime)
        report["asset_verification_seconds"] = time.perf_counter() - asset_start
        report["checks"]["pinned_assets"] = True
        with tempfile.TemporaryDirectory(prefix="grail_graspgen_empty_gripper_") as gripper_dir:
            backend = _load_backend(runtime, gripper_dir)
            report["versions"] = _versions()
            report["gpu"] = backend[1].cuda.get_device_name()
            report["checks"].update(imports=True, cuda=True)
            if request["action"] == "run":
                _generate(request, runtime, output, backend, report)
            else:
                report["status"] = "PREFLIGHT_PASSED"
        report["ok"] = True
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
        report["status"] = "PREFLIGHT_FAILED" if isinstance(request, dict) and request.get("action") == "preflight" else "FAILED"
    report["elapsed_seconds"] = time.perf_counter() - started
    if output_created:
        _json(output / "worker_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        # Upstream log/print output is diagnostics; stdout is one JSON receipt.
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(request)
    except Exception as exc:
        result = dict(ok=False, status="FAILED", selected=False, checks={}, errors=[type(exc).__name__ + ": " + str(exc)])
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
