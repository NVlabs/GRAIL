"""Explicit local GraspGenX proposals; never edit or select a native motion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required input is missing: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def _path(value, base, *, executable=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty file or directory path")
    path = Path(value).expanduser()
    if executable and "/" not in value:
        found = shutil.which(value)
        if found is None:
            raise ValueError(f"Python executable is not on PATH: {value}")
        return found
    path = path if path.is_absolute() else base / path
    # Resolving the interpreter symlink would discard its virtual environment.
    return os.path.abspath(path) if executable else str(path.resolve())


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _timeout(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise ValueError("timeout must be a finite positive number")


def plan(config_path, output):
    """Bind configuration and files without importing a model or running a worker."""
    config_path = Path(config_path).expanduser().resolve()
    cfg = json.loads(config_path.read_text(encoding="utf8"))
    if not isinstance(cfg, dict):
        raise ValueError("GraspGenX configuration must be a JSON object")
    allowed = {"native", "model_xml", "object_mesh", "hand_profile", "side", "frame",
               "point_count", "seed", "num_grasps", "batch_size", "runtime", "preview",
               "label", "expected_input_sha256"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown GraspGenX configuration fields: {sorted(unknown)}")
    if cfg.get("hand_profile") != "g1_dex3":
        raise ValueError("Explicit hand_profile='g1_dex3' is required")
    if cfg.get("side") not in ("left", "right"):
        raise ValueError("Explicit side='left' or 'right' is required")
    cfg["frame"] = _integer(cfg.get("frame"), "frame", 0, 2**31 - 1)
    for name, default, low, high in (("point_count", 4096, 32, 65536),
                                    ("seed", 20260929, 0, 2**32 - 1),
                                    ("num_grasps", 2000, 1, 10000),
                                    ("batch_size", 200, 1, 2000)):
        cfg[name] = _integer(cfg.get(name, default), name, low, high)
    cfg["preview"] = cfg.get("preview", True)
    if type(cfg["preview"]) is not bool:
        raise ValueError("preview must be boolean")
    if "label" in cfg and (not isinstance(cfg["label"], str) or len(cfg["label"]) > 1000):
        raise ValueError("label must be a string of at most 1000 characters")
    base = config_path.parent
    bindings = {"config": _binding(config_path)}
    for key in ("native", "model_xml", "object_mesh"):
        cfg[key] = _path(cfg.get(key), base)
        bindings[key] = _binding(cfg[key])
    expected = cfg.get("expected_input_sha256", {})
    if not isinstance(expected, dict) or set(expected) - {"native", "model_xml", "object_mesh"}:
        raise ValueError("expected_input_sha256 must map known input names to hashes")
    for key, digest in expected.items():
        if digest != bindings[key]["sha256"]:
            raise ValueError(f"Expected input hash differs: {key}")
    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) - {"python", "repo", "checkpoint_dir", "device"}:
        raise ValueError("runtime must configure python, repo, checkpoint_dir and optional device")
    cfg["runtime"] = {key: _path(runtime.get(key), base, executable=key == "python")
                      for key in ("python", "repo", "checkpoint_dir")}
    cfg["runtime"]["device"] = runtime.get("device", "cuda:0")
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output must be a new directory: {output}")
    output = output.resolve()
    for key in ("repo", "checkpoint_dir"):
        protected = Path(cfg["runtime"][key])
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output overlaps the external {key}: {protected}")
    code = {name: _binding(Path(__file__).with_name(name)) for name in
            ("graspgen.py", "graspgen_inputs.py", "graspgen_runtime.py", "graspgen_worker.py", "graspgen_preview.py")}
    return {"schema_version": 1, "status": "PLANNED_GRASP_PROPOSALS", "selected": False,
            "config": cfg, "bindings": bindings, "implementation": code,
            "output": str(output), "default_workflow_enabled": False,
            "scope": "Object-relative grasp proposals at one native reference frame; no joint edits, IK or motion selection."}


def _verify_plan(proposal):
    fresh = plan(proposal["bindings"]["config"]["path"], proposal["output"])
    for key in ("schema_version", "config", "bindings", "implementation", "output"):
        if proposal.get(key) != fresh[key]:
            raise ValueError(f"GraspGenX plan changed or is stale: {key}")


def preflight(proposal, *, timeout=120):
    """Validate native geometry and the external runtime without generating grasps."""
    from . import graspgen_inputs, graspgen_runtime

    _timeout(timeout)
    _verify_plan(proposal)
    checks = {}
    try:
        with tempfile.TemporaryDirectory(prefix="retargeter_graspgen_inputs_") as folder:
            inputs = graspgen_inputs.prepare_inputs(proposal["config"], Path(folder))
            graspgen_inputs.verify_bindings(inputs)
        checks["inputs"] = {"ok": True}
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        checks["inputs"] = {"ok": False, "error": str(error)}
    # A malformed model should not initiate heavyweight runtime work.
    if checks["inputs"]["ok"]:
        checks["runtime"] = graspgen_runtime.preflight(proposal["config"]["runtime"], timeout=timeout)
    ok = all(check.get("ok") is True for check in checks.values())
    return {"schema_version": 1, "status": "READY_FOR_EXPLICIT_RUN" if ok else "BLOCKED",
            "ok": ok, "selected": False, "checks": checks, "generation_performed": False}


def execute(proposal, *, timeout=None):
    """Generate candidates through one foreground external worker, then preview them."""
    import numpy as np
    from . import graspgen_inputs, graspgen_runtime, graspgen_preview

    _timeout(timeout)
    _verify_plan(proposal)
    start = time.perf_counter()
    cfg = proposal["config"]
    output = Path(proposal["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "RUNNING", "selected": False,
              "plan": proposal, "motion_modified": False}
    _write(output / "run.json", report)
    try:
        inputs_report = graspgen_inputs.prepare_inputs(cfg, output)
        prepared_bindings = {name: _binding(output / name) for name in ("input.npz", "hand_description.json", "inputs.json")}
        request = {"action": "run", "runtime": cfg["runtime"],
                   "input": str(output / "input.npz"), "descriptor": str(output / "hand_description.json"),
                   "output": str(output / "model"), "num_grasps": cfg["num_grasps"],
                   "batch_size": cfg["batch_size"], "seed": cfg["seed"]}
        _write(output / "request.json", request)
        worker = graspgen_runtime.run_worker(output / "request.json", cfg["runtime"], timeout=timeout)
        if worker.get("ok") is not True or worker.get("selected") is not False or worker.get("status") != "PROPOSALS_UNREVIEWED":
            raise RuntimeError(f"GraspGenX worker did not complete: {worker}")
        graspgen_inputs.verify_bindings(inputs_report)
        for item in [*proposal["bindings"].values(), *proposal["implementation"].values(), *prepared_bindings.values()]:
            if _sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Bound input changed during generation: {item['path']}")
        with np.load(output / "model/proposals.npz", allow_pickle=False) as archive:
            grasps, confidence = archive["grasps_object"], archive["confidence"]
        if len(grasps) != cfg["num_grasps"]:
            raise ValueError("GraspGenX returned a different proposal count")
        with np.load(output / "input.npz", allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in archive.files}
        derived = graspgen_inputs.validate_proposals(grasps, confidence, inputs)
        with (output / "proposals.npz").open("xb") as stream:
            np.savez_compressed(stream, grasps_object=grasps, confidence=np.asarray(confidence).reshape(-1), **derived)
        report.update(status="GRASP_PROPOSALS_UNREVIEWED", count=len(grasps),
                      side=cfg["side"], frame=cfg["frame"], label=cfg.get("label", "GraspGenX proposals"),
                      inputs=inputs_report, prepared_bindings=prepared_bindings, worker=worker,
                      proposals=_binding(output / "proposals.npz"),
                      reference_semantics="Current native reference pose; may already be refined. It is not an observed human hand or a contact target.",
                      checks={"native_input_binding": True, "finite_rigid_proposals": True,
                              "mounted_ik": "NOT_RUN", "collision": "NOT_RUN", "contact": "NOT_RUN",
                              "temporal": "NOT_RUN"})
        if cfg["preview"]:
            preview = graspgen_preview.write_preview(output, output / "input.npz", output / "proposals.npz", report)
            report["preview"] = _binding(preview)
        report["total_seconds"] = time.perf_counter() - start
        _write(output / "run.json", report)
    except BaseException as error:
        report.update(status="FAILED_UNSELECTED", error=f"{type(error).__name__}: {error}",
                      total_seconds=time.perf_counter() - start)
        _write(output / "run.json", report)
        raise
    return {"status": report["status"], "selected": False, "motion_modified": False,
            "count": report["count"], "report": str(output / "run.json"),
            "proposals": str(output / "proposals.npz"),
            "preview": report.get("preview", {}).get("path")}


def add_cli(subparsers):
    parser = subparsers.add_parser("graspgenx", help="On-demand local grasp proposals; never edits or selects motion")
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--config", required=True, help="Native frame, original geometry and external runtime JSON")
    parser.add_argument("--output", required=True, help="New proposal directory")
    parser.add_argument("--timeout", type=float, default=None, help="External process timeout in seconds")
    return parser


def run_cli(args):
    proposal = plan(args.config, args.output)
    if args.action == "plan":
        return proposal
    if args.action == "preflight":
        return preflight(proposal, timeout=args.timeout if args.timeout is not None else 120)
    return execute(proposal, timeout=args.timeout)


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_cli(sub)
    args = parser.parse_args(["graspgenx", *(sys.argv[1:] if argv is None else argv)])
    try:
        result = run_cli(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"GraspGenX error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if args.action == "preflight" and not result["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
