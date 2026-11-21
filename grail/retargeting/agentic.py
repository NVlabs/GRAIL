"""Delegate explicit GRAIL model handoffs to the pinned Agentic Retargeter.

All policy and paths live in YAML. Each invocation writes a new request/receipt
folder; conversion and authored workflow commands stay in foreground children.
The external checkout owns IK, robot editing, screening and timed stage replay.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

PATH_FIELDS = {
    "grail_root",
    "soma_root",
    "source",
    "handoff",
    "skeleton_meta",
    "contact_constraints",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty path string")
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _python(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit Python executable is required")
    found = shutil.which(value) if "/" not in value else str(_path(value, base))
    if not found or not Path(found).is_file() or not os.access(found, os.X_OK):
        raise ValueError(f"Python executable not found: {value}")
    # Do not resolve venv interpreter symlinks to the system Python.
    return found


def _write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _runtime(cfg, base):
    cfg = _mapping(cfg, "runtime")
    root = _path(cfg.get("checkout", str(ROOT / "imports/agentic-retargeter")), base)
    python = _python(cfg.get("python"), base)
    paths = ["tools/retarget/run.py", "tools/retarget/native.py", "tools/batch/run.py"]
    for name in paths:
        if not (root / name).is_file():
            raise ValueError(f"Agentic Retargeter checkout is missing {name}: {root}")
    vendored = root / "VENDORED_REVISION"
    if not (root / ".git").exists() and vendored.is_file():
        # The in-repository copy is a plain vendored snapshot; Git HEAD would
        # describe the enclosing GRAIL checkout, not the retargeter revision.
        actual = vendored.read_text().strip() or None
    else:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        actual = revision.stdout.strip() if revision.returncode == 0 else None
    if cfg.get("revision") is not None and actual != cfg["revision"]:
        raise ValueError("Agentic Retargeter revision differs from runtime.revision")
    files = {str(p.relative_to(root)): _binding(p) for p in sorted((root / "src").rglob("*.py"))}
    files.update({name: _binding(root / name) for name in paths})
    return {"checkout": str(root), "python": python, "revision": actual, "code": files}


def load_config(path):
    """Resolve YAML once, with explicit model routing and no inferred fallback."""
    path = Path(path).expanduser().resolve(strict=True)
    cfg = _mapping(yaml.safe_load(path.read_text()), "configuration")
    if cfg.get("schema_version") != 1 or cfg.get("source_model") != "soma03":
        raise ValueError("Use schema_version: 1 and source_model: soma03")
    allowed = {
        "schema_version",
        "source_model",
        "runtime",
        "backend",
        "source",
        "native",
        "workflow",
        "timeout_seconds",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = copy.deepcopy(cfg)
    cfg["config"] = _binding(path)
    cfg["runtime"] = _runtime(cfg.get("runtime"), path.parent)
    if "timeout_seconds" in cfg:
        _positive(cfg["timeout_seconds"], "timeout_seconds")
    backend = _mapping(cfg.get("backend", {}), "backend")
    backend["human_scale_ratio"] = float(
        _positive(backend.get("human_scale_ratio", 1.0), "backend.human_scale_ratio")
    )
    for key in PATH_FIELDS & backend.keys():
        backend[key] = str(_path(backend[key], path.parent))
    if "extra_robot_paths" in backend:
        paths = backend["extra_robot_paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("backend.extra_robot_paths must be a nonempty list of paths")
        # Upstream's robot registry consumes directory paths. Resolve these
        # against the user's YAML before writing a backend JSON elsewhere.
        backend["extra_robot_paths"] = [str(_path(value, path.parent)) for value in paths]
    if "python" in backend:
        backend["python"] = _python(backend["python"], path.parent)
    if "human_model" in backend:
        raise ValueError("Bind native SOMA identity in source.model_config")
    if "bindings" in backend:
        backend["bindings"] = {
            key: str(_path(value, path.parent))
            for key, value in _mapping(backend["bindings"], "backend.bindings").items()
        }
    cfg["backend"] = backend
    if "source" in cfg:
        source = _mapping(cfg["source"], "source")
        if "source" in backend:
            raise ValueError("Choose raw source.path or backend.source BVH, not both")
        source["path"] = str(_path(source["path"], path.parent))
        source["binding"] = _binding(source["path"])
        source["python"] = _python(source.get("python"), path.parent)
        _mapping(source.get("model_config"), "source.model_config")
        source["model_config"] = _model_paths(source["model_config"], path.parent)
        source["ground_initial_frame"] = source.get("ground_initial_frame", True)
        if type(source["ground_initial_frame"]) is not bool:
            raise ValueError("source.ground_initial_frame must be boolean")
        if source["ground_initial_frame"]:
            # These inputs were built in another world frame. Fresh targets must
            # be authored from the grounded export rather than mixed with it.
            external_targets = {
                "contact_constraints",
                "robot_hand_targets",
                "wrist_keyframes",
                "pose_keyframes",
                "prior_global_seed",
            } & backend.keys()
            temporal = backend.get("temporal_optimization", {})
            if isinstance(temporal, dict) and "prior_global_seed" in temporal:
                external_targets.add("temporal_optimization.prior_global_seed")
            if external_targets:
                raise ValueError(
                    "Ground the raw source before preparing external world targets: "
                    + ", ".join(sorted(external_targets))
                    + ". Build them from the grounded export, then use backend.source BVH."
                )
    native = _mapping(cfg.get("native", {}), "native")
    if type(native.get("enabled", True)) is not bool:
        raise ValueError("native.enabled must be boolean")
    for key in ("object_sidecar", "object_mesh", "scene_sidecar"):
        if key in native:
            native[key] = str(_path(native[key], path.parent))
    if "object_mesh" in native and "source" not in cfg:
        raise ValueError(
            "native.object_mesh requires raw SOMA source export; use object_sidecar for a pre-exported BVH"
        )
    if "object_mesh" in native and "object_sidecar" in native:
        raise ValueError("Choose native.object_mesh or native.object_sidecar, not both")
    if cfg.get("source", {}).get("ground_initial_frame") and (
        "object_sidecar" in native
        or (backend.get("bindings", {}).get("object") and "object_mesh" not in native)
    ):
        raise ValueError(
            "Grounded raw source requires native.object_mesh to regenerate the object/scene "
            "sidecar; an existing object binding may use the original world frame"
        )
    if "scene_sidecar" in native and "object_mesh" not in native:
        raise ValueError(
            "SOMA scene_sidecar requires raw source and object_mesh; for BVH embed the scene in object_sidecar"
        )
    cfg["native"] = native
    if "workflow" in cfg:
        flow = _mapping(cfg["workflow"], "workflow")
        flow["plan"] = str(_path(flow["plan"], path.parent))
        workers = flow.get("workers", 1)
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workflow.workers must be an integer between 1 and 32")
        flow["workers"] = workers
        if type(flow.get("resume", False)) is not bool:
            raise ValueError("workflow.resume must be boolean")
        if "output" in flow:
            flow["output"] = str(_path(flow["output"], path.parent))
        if flow.get("resume") and "output" not in flow:
            raise ValueError("workflow.resume requires the previous workflow.output")
    return cfg


def _model_paths(value, base):
    """Resolve model asset settings while preserving opaque native model policy."""
    result = copy.deepcopy(value)
    for key, item in result.items():
        if isinstance(item, dict):
            result[key] = _model_paths(item, base)
        elif isinstance(item, str) and (
            key.endswith("_path") or key.endswith("_root") or key == "gem_x_soma_assets"
        ):
            result[key] = str(_path(item, base))
    return result


def _invoke(argv, output, name, *, cwd, timeout=None, env=None):
    start = time.monotonic()
    row = {"name": name, "argv": argv, "cwd": str(cwd)}
    log = output / f"{name}.log"
    row["log"] = str(log)
    with log.open("x", encoding="utf8") as stream:
        process = None
        group_args = (
            {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
        )
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **group_args,
            )
            code = process.wait(timeout=timeout)
            row.update(returncode=code, status="EXECUTED" if code == 0 else "FAILED")
        except subprocess.TimeoutExpired:
            row.update(returncode=None, status="TIMED_OUT")
        except OSError as error:
            row.update(returncode=None, status="FAILED", error=str(error))
        finally:
            if process is not None:
                # A router may spawn Blender/the solver. Own and reap the entire
                # foreground process group even when the router alone exits.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    row["seconds"] = time.monotonic() - start
    row["log_sha256"] = _sha(log)
    _write(output / f"{name}.receipt.json", row)
    return row


def _json_log(row):
    text = Path(row["log"]).read_text()
    try:
        return json.loads(text)
    except ValueError:
        return None


def _check_runtime(runtime):
    for value in runtime["code"].values():
        if _sha(value["path"]) != value["sha256"]:
            raise RuntimeError("Agentic Retargeter code changed during this invocation")


def execute(config_path, output, *, action="run"):
    """Execute a versioned delegation; success never accepts a robot motion.

    ``plan`` and ``preflight`` only prepare a source export request for raw SOMA.
    ``run`` performs that export before the official BVH/runtime preflight.
    ``workflow`` runs an authored native-tool plan, independently of conversion.
    """
    if action not in {"plan", "preflight", "run", "workflow"}:
        raise ValueError(f"Unsupported action: {action}")
    cfg = load_config(config_path)
    output = Path(output).expanduser().resolve()
    root = Path(cfg["runtime"]["checkout"])
    for protected in (root, ROOT):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output must be outside code/config input trees: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "request.json", cfg)
    report = {
        "schema_version": 1,
        "source_model": cfg["source_model"],
        "backend": "soma",
        "action": action,
        "config": cfg["config"],
        "runtime": cfg["runtime"],
        "bridge_code": {
            p.name: _binding(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("soma_export.py"),
                Path(__file__).with_name("handoff.py"),
                Path(__file__).with_name("scene.py"),
                Path(__file__).with_name("grounding.py"),
            ]
            if p.is_file()
        },
        "motion_status": "UNREVIEWED",
        "selected": False,
        "stages": [],
        "status": "PREPARING_UNREVIEWED",
        "outputs": {},
    }
    started = time.monotonic()
    try:
        if action == "workflow":
            _workflow(cfg, output, report)
        else:
            _convert(cfg, output, report)
        _check_runtime(cfg["runtime"])
        for item in report["bridge_code"].values():
            if _sha(item["path"]) != item["sha256"]:
                raise RuntimeError("GRAIL bridge code changed during this invocation")
        if _sha(cfg["config"]["path"]) != cfg["config"]["sha256"]:
            raise RuntimeError("Configuration changed during execution")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        report.update(status="FAILED_UNREVIEWED", error=str(error))
    finally:
        for name in ("run.json", "retarget_result.json"):
            candidate = output / "retarget" / name
            if candidate.is_file():
                report.setdefault("retained_backend_reports", {})[name] = _binding(candidate)
        report["seconds"] = time.monotonic() - started
        _write(output / "bridge_result.json", report)
    return report


def _stage(cfg, output, report, argv, name, *, cwd=None, env=None):
    row = _invoke(
        argv,
        output,
        name,
        cwd=cwd or cfg["runtime"]["checkout"],
        timeout=cfg.get("timeout_seconds"),
        env=env,
    )
    report["stages"].append(row)
    if row["status"] != "EXECUTED":
        report["status"] = "FAILED_UNREVIEWED"
        return None
    return row


def _convert(cfg, output, report):
    backend = copy.deepcopy(cfg["backend"])
    if "source" in cfg:
        model_file = output / "source_model.json"
        _write(model_file, cfg["source"]["model_config"])
        export_dir = output / "source_export"
        argv = [
            cfg["source"]["python"],
            "-m",
            "grail.retargeting.soma_export",
            "--source",
            cfg["source"]["path"],
            "--output",
            str(export_dir),
            "--model-config",
            str(model_file),
            "--soma-root",
            backend["soma_root"],
        ]
        if backend.get("python"):
            argv += ["--upstream-python", backend["python"]]
        if cfg["source"]["ground_initial_frame"]:
            argv += ["--ground-initial-frame"]
        report["source_export"] = {"argv": argv, "source": cfg["source"]["binding"]}
        if report["action"] != "run":
            report.update(status="SOURCE_EXPORT_REQUIRED_UNREVIEWED", ready=False)
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        if _stage(cfg, output, report, argv, "source_export", cwd=ROOT, env=env) is None:
            return
        manifest_path = export_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if cfg["source"]["ground_initial_frame"] and not manifest.get("grounding", {}).get(
            "enabled"
        ):
            raise ValueError("Requested initial grounding is missing from source export receipt")
        backend.update(
            source=str(export_dir / "source.bvh"),
            source_units=manifest["source_units"],
            source_facing_direction=manifest["source_facing_direction"],
            fps=manifest["fps"],
        )
        backend.setdefault("bindings", {})["human_model"] = str(manifest_path)
        report["outputs"]["source_export"] = _binding(manifest_path)
        report["grounding"] = manifest.get("grounding", {"enabled": False})
        if cfg["source"]["ground_initial_frame"] and backend["bindings"].get("camera"):
            from .grounding import translate_camera

            camera_binding = _binding(backend["bindings"]["camera"])
            camera = json.loads(Path(camera_binding["path"]).read_text())
            shifted = translate_camera(camera, manifest["world_offset"])
            shifted["grounding_parent_camera"] = camera_binding
            if _binding(camera_binding["path"]) != camera_binding:
                raise ValueError("Camera changed during grounding")
            camera_path = output / "grounded_camera.json"
            _write(camera_path, shifted)
            backend["bindings"]["camera"] = str(camera_path)
            report["outputs"]["camera"] = _binding(camera_path)
        if cfg["native"].get("object_mesh"):
            sidecar = output / "object_sidecar.npz"
            request = {
                "hoi_path": cfg["source"]["path"],
                "output_path": str(sidecar),
                "bvh_path": backend["source"],
                "export_manifest": str(manifest_path),
                "object_mesh_path": cfg["native"]["object_mesh"],
                "object_scale": cfg["native"].get("object_scale"),
                "scene_sidecar": cfg["native"].get("scene_sidecar"),
            }
            request_path = output / "object_sidecar_request.json"
            _write(request_path, request)
            argv = [
                cfg["source"]["python"],
                "-m",
                "grail.retargeting.handoff",
                "--request",
                str(request_path),
            ]
            if _stage(cfg, output, report, argv, "object_sidecar", cwd=ROOT, env=env) is None:
                return
            cfg["native"]["object_sidecar"] = str(sidecar)
            backend["bindings"]["object"] = str(sidecar)
            report["outputs"]["object_sidecar"] = _binding(sidecar)
    backend_config = output / "backend.json"
    _write(backend_config, backend)
    runtime = cfg["runtime"]
    prefix = [runtime["python"], str(Path(runtime["checkout"]) / "tools/retarget/run.py")]
    common = [
        "--source-model",
        "soma",
        "--config",
        str(backend_config),
        "--output",
        str(output / "retarget"),
    ]
    action = report["action"]
    if action in {"preflight", "run"}:
        row = _stage(cfg, output, report, [*prefix, "preflight", *common], "preflight")
        if row is None:
            return
        result = _json_log(row)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("Backend preflight did not report ok=true")
        if action == "preflight":
            report.update(status="READY_UNREVIEWED", backend_result=result)
            return
    argv = [*prefix, action, *common]
    if action == "run" and cfg.get("timeout_seconds") is not None:
        argv += ["--timeout", str(cfg["timeout_seconds"])]
    row = _stage(cfg, output, report, argv, action)
    if row is None:
        return
    if action == "plan":
        result = _json_log(row)
        if not isinstance(result, dict) or result.get("source_model") != "soma":
            raise RuntimeError("Malformed delegated retarget plan")
        report.update(status=result["status"], backend_result=result)
        return
    run_report = output / "retarget/run.json"
    unified = output / "retarget/retarget_result.json"
    result = json.loads(unified.read_text())
    if result.get("source_model") != "soma":
        raise RuntimeError("Retarget source model changed")
    report.update(status=result["status"], backend_result=result)
    report["outputs"].update(run=_binding(run_report), retarget=_binding(unified))
    if result["status"] != "CONVERTED_UNREVIEWED":
        return
    if cfg["native"].get("enabled", True):
        native = output / "native/robot_UNREVIEWED.npz"
        native.parent.mkdir()
        argv = [
            runtime["python"],
            str(Path(runtime["checkout"]) / "tools/retarget/native.py"),
            "--report",
            str(run_report),
            "--output",
            str(native),
        ]
        if cfg["native"].get("object_sidecar"):
            argv += ["--object-sidecar", cfg["native"]["object_sidecar"]]
        if _stage(cfg, output, report, argv, "native") is None:
            return
        report["outputs"].update(
            native=_binding(native), native_report=_binding(str(native) + ".json")
        )
        report["native_result"] = json.loads(Path(str(native) + ".json").read_text())


def _workflow(cfg, output, report):
    flow = cfg.get("workflow")
    if not flow:
        raise ValueError("workflow action requires workflow.plan")
    plan_path = Path(flow["plan"])
    plan = _mapping(yaml.safe_load(plan_path.read_text()), "workflow.plan")
    for case in plan.get("cases", []):
        case["cwd"] = str(_path(case.get("cwd", "."), plan_path.parent))
    materialized = output / "workflow.json"
    _write(materialized, plan)
    report["workflow_plan"] = _binding(plan_path)
    runtime = cfg["runtime"]
    budget_file = Path(runtime["checkout"]) / "src/case_budget.py"
    if budget_file.is_file():
        report["workflow_budget_policy"] = _budget_policy(budget_file)
    workflow_output = Path(flow.get("output", str(output / "workflow")))
    argv = [
        runtime["python"],
        str(Path(runtime["checkout"]) / "tools/batch/run.py"),
        "--plan",
        str(materialized),
        "--output",
        str(workflow_output),
        "--workers",
        str(flow["workers"]),
    ]
    if flow.get("resume"):
        argv.append("--resume")
        report["resumed_budget_bindings"] = {
            str(p): _binding(p) for p in sorted(workflow_output.glob("*/case_budget.json"))
        }
    row = _stage(cfg, output, report, argv, "workflow")
    result_path = workflow_output / "execution.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        report["outputs"]["workflow"] = _binding(result_path)
        report["workflow_result"] = result
        cases = result.get("cases", [])
        complete = cases and all(
            case.get("status") == "EXECUTION_COMPLETE_UNREVIEWED" for case in cases
        )
        if row is not None and complete:
            report["status"] = "EXECUTION_COMPLETE_UNREVIEWED"
        else:
            report["status"] = "INCOMPLETE_UNREVIEWED"
    elif row is not None:
        raise RuntimeError("Workflow returned without execution.json")


def _budget_policy(path):
    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return number(node.left) * number(node.right)
        raise ValueError("Nonliteral budget policy")

    policy = {
        "source": _binding(path),
        "task_completion": False,
        "note": "The external per-case budget and delivery reserve bound execution, not requested motion acceptance. Resume the same batch output to retain its ledger.",
    }
    names = {
        "CASE_LIMIT_SECONDS": "case_seconds",
        "DELIVERY_RESERVE_SECONDS": "delivery_reserve_seconds",
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        policy[names[target.id]] = number(node.value)
                    except ValueError:
                        pass
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run", "workflow"))
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--output", required=True, help="New request/receipt directory")
    args = parser.parse_args(argv)
    try:
        result = execute(args.config, args.output, action=args.action)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Agentic retargeting: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    failed = result["status"] in {"FAILED_UNREVIEWED", "INCOMPLETE_UNREVIEWED", "BLOCKED"}
    failed = failed or (args.action == "preflight" and result["status"] != "READY_UNREVIEWED")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
