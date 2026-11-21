"""Bound, foreground adapter for an external GRAIL/GMR installation.

No model loading or inference happens in this package. The source handoff must
bind the same human model assets used by the external loader before conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _tree(path: Path) -> dict[str, dict[str, str]]:
    files = {str(p.relative_to(path)): _file(p) for p in sorted(path.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if not files:
        raise ValueError(f"Required asset directory is empty or missing: {path}")
    return files


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _path(value: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _hashes(files: dict[str, dict[str, str]]) -> dict[str, str]:
    return {name: item["sha256"] for name, item in files.items()}


def plan(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Read and bind inputs; return commands and blockers without executing them.

    ``required_handoff`` describes the evidence a reviewed upstream handoff must
    supply. Its presence in a plan is *not* a review or model-equivalence claim.
    """
    config_path = Path(config_path).resolve()
    cfg = _read(config_path)
    base = config_path.parent
    root = _path(cfg["grail_root"], base)
    source = _path(cfg["source"], base)
    if source.name != "hoi_data.pkl" or source.parent.name != "hoi_data":
        raise ValueError("Expected upstream <case>/hoi_data/hoi_data.pkl (GRAB NPZ is not supported here)")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output must be a new path: {output}")
    model_cfg = cfg.get("human_model", {})
    convention = model_cfg.get("convention", "custom_g1")
    if convention not in {"custom_g1", "standard"}:
        raise ValueError("human_model.convention must be custom_g1 or standard")
    body_models = _path(model_cfg["body_models"], base)
    for protected in (root, source.parent.parent, body_models):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output overlaps a protected input/upstream tree: {protected}")
    python_value = cfg.get("python", "python")
    if not isinstance(python_value, str):
        raise ValueError("python must be an executable path, not a shell command")
    # Preserve the virtualenv launcher path instead of resolving it to the
    # system interpreter and silently losing the selected environment.
    python_path = Path(python_value).expanduser()
    python = (shutil.which(python_value) if "/" not in python_value else
              os.path.abspath(python_path if python_path.is_absolute() else base / python_path))
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python executable not found: {python_value}")
    fps = cfg.get("fps", 24)
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer matching the source")
    robot = cfg.get("robot", "unitree_g1_with_hands")
    if robot not in {"unitree_g1", "unitree_g1_with_hands"}:
        raise ValueError("Unsupported GMR robot")
    process_hands = cfg.get("process_hands", False)
    if type(process_hands) is not bool:
        raise ValueError("process_hands must be boolean")

    # Hash all base model assets, not just a convenient template. The loader
    # also consumes skinning weights, shape/pose directions and hand bases.
    human_assets = {f"body_models/{name}": entry for name, entry in _tree(body_models).items()}
    if convention == "custom_g1":
        for name in ("g1_smplx_tpose.obj", "g1_smplx_param.npz"):
            human_assets[f"custom/{name}"] = _file(root / "data/body_profiles/unitree_g1/smplx" / name)
    source_binding = _file(source)
    mesh_assets = _tree(source.parent.parent / "mesh_data")
    if "model.obj" not in mesh_assets:
        raise ValueError("Source mesh_data/model.obj is required by the upstream loader")
    code_paths = ["grail/retargeting/retarget.py", "grail/models/g1_smplx_model.py",
                  "grail/adapters/gmr.py", "imports/GMR/general_motion_retargeting/params.py"]
    if process_hands:
        code_paths.append("grail/retargeting/process.py")
    code = {name: _file(root / name) for name in code_paths}
    robot_assets = _tree(root / "imports/GMR/assets/unitree_g1")
    robot_xml = "g1_mocap_29dof_with_hands.xml" if robot.endswith("with_hands") else "g1_mocap_29dof.xml"
    if robot_xml not in robot_assets:
        raise ValueError(f"Missing native robot model: {robot_xml}")
    ik_config = _file(root / "data/body_profiles/unitree_g1/smplx/gmr_smplx_to_g1.json")
    required_handoff = {
        "schema_version": 1,
        "source_sha256": source_binding["sha256"],
        "fps": fps,
        "human_model": {"convention": convention, "asset_sha256": _hashes(human_assets),
                        "loader_sha256": code["grail/models/g1_smplx_model.py"]["sha256"]},
    }
    blockers = []
    handoff_binding = None
    if cfg.get("handoff"):
        handoff_path = _path(cfg["handoff"], base)
        handoff = _read(handoff_path)
        handoff_binding = _file(handoff_path)
        for key in required_handoff:
            if handoff.get(key) != required_handoff[key]:
                blockers.append(f"Upstream handoff mismatch: {key}")
    else:
        blockers.append("Missing upstream handoff: source, FPS and actual human loader assets must be bound")
    commands = [[python, "-u", "-m", "grail.retargeting.retarget", "--file", str(source),
                 "--output_dir", str(output / "raw"), "--smplx_model_path", str(body_models),
                 "--robot", robot, "--fps", str(fps), "--no_viewer"]]
    if convention == "standard":
        commands[0].append("--no_g1_proportions")
    skeleton = None
    if process_hands:
        skeleton = _file(_path(cfg.get("skeleton_meta", str(root / "data/body_profiles/unitree_g1/smplx/g1_skeleton_meta.pkl")), base))
        commands.append([python, "-u", "-m", "grail.retargeting.process", "--input", str(output / "raw"),
                         "--output", str(output / "processed"), "--meta_pkl", skeleton["path"],
                         "--treat_hands_equally", "--include_contact_points", "--per_object",
                         "--grasp_anticipation_frames", "0"])
    return {
        "schema_version": 1, "status": "BLOCKED" if blockers else "PLANNED_UNREVIEWED",
        "config": _file(config_path), "cwd": str(root), "output": str(output), "commands": commands,
        "source": source_binding, "source_mesh_assets": mesh_assets, "human_model_assets": human_assets,
        "upstream_code": code, "robot_assets": robot_assets, "ik_config": ik_config,
        "skeleton_meta": skeleton, "handoff": handoff_binding, "required_handoff": required_handoff,
        "blockers": blockers,
        "limitations": ["This adapter uses the upstream baked-model loader; arbitrary custom model snapshots are not supported.",
                        "Handoff model equivalence is an upstream review responsibility; hashing does not establish it.",
                        "Initial conversion and optional hand-action processing do not validate contact or accept motion."],
    }


def _environment(root: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Make the configured checkout authoritative for grail, while preserving
    # the external environment's installed GMR and heavy dependencies.
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def preflight(proposal: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    """Run external CLI help to test imports and supported options; no conversion."""
    checks = []
    for command in proposal["commands"]:
        argv = command[:4] + ["--help"]
        try:
            result = subprocess.run(argv, cwd=proposal["cwd"], env=_environment(proposal["cwd"]),
                                    capture_output=True, text=True, timeout=timeout, check=False)
            expected = ("--file", "--no_g1_proportions", "--no_viewer") if command[3].endswith("retarget") else (
                "--treat_hands_equally", "--include_contact_points", "--grasp_anticipation_frames")
            missing = [flag for flag in expected if flag not in result.stdout]
            checks.append({"command": argv, "returncode": result.returncode, "missing_options": missing,
                           "ok": result.returncode == 0 and not missing,
                           "diagnostic": result.stderr[-4000:] if result.returncode else ""})
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append({"command": argv, "ok": False, "diagnostic": str(exc)})
    # A different installed GMR can resolve different robot XML paths. Refuse
    # that situation instead of claiming hashes of an unused local asset tree.
    argv = [proposal["commands"][0][0], "-c",
            "import importlib.util,json;print(json.dumps({'gmr':importlib.util.find_spec('general_motion_retargeting').origin}))"]
    expected_gmr = Path(proposal["cwd"]) / "imports/GMR/general_motion_retargeting/__init__.py"
    try:
        result = subprocess.run(argv, cwd=proposal["cwd"], env=_environment(proposal["cwd"]),
                                capture_output=True, text=True, timeout=timeout, check=False)
        actual = json.loads(result.stdout).get("gmr") if result.returncode == 0 else None
        checks.append({"command": argv, "ok": bool(actual) and Path(actual).resolve() == expected_gmr.resolve(),
                       "resolved_gmr": actual, "expected_gmr": str(expected_gmr),
                       "diagnostic": result.stderr[-4000:] if result.returncode else ""})
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
        checks.append({"command": argv, "ok": False, "diagnostic": str(exc)})
    return {"schema_version": 1, "ok": all(c["ok"] for c in checks) and not proposal["blockers"],
            "blockers": proposal["blockers"], "checks": checks}


def _bound_files(proposal: dict[str, Any]) -> list[dict[str, str]]:
    items = [proposal["config"], proposal["source"], proposal["ik_config"]]
    for key in ("source_mesh_assets", "human_model_assets", "upstream_code", "robot_assets"):
        items.extend(proposal[key].values())
    items.extend(proposal[key] for key in ("handoff", "skeleton_meta") if proposal[key])
    return items


def _assert_unchanged(proposal: dict[str, Any]) -> None:
    for entry in _bound_files(proposal):
        path = Path(entry["path"])
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise ValueError(f"Bound input changed: {path}")


def execute(proposal: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Reserve a new output directory and run each external stage in foreground."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive and finite")
    if proposal["blockers"]:
        raise ValueError("; ".join(proposal["blockers"]))
    checks = preflight(proposal)
    if not checks["ok"]:
        raise ValueError("External GMR preflight failed: " + json.dumps(checks))
    _assert_unchanged(proposal)
    output = Path(proposal["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(proposal, status="RUNNING_UNREVIEWED", preflight=checks, stages=[])
    report_path = output / "run.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for index, command in enumerate(proposal["commands"]):
            log = output / f"stage_{index + 1:02d}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=proposal["cwd"], env=_environment(proposal["cwd"]),
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            report["stages"].append({"command": command, "returncode": result.returncode, "log": log.name})
            save()
            if result.returncode:
                raise RuntimeError(f"GMR stage {index + 1} exited {result.returncode}; see {log}")
            stage_dir = output / ("raw" if index == 0 else "processed")
            for folder in ("robot", "objects"):
                if not any((stage_dir / folder).glob("*.pkl")):
                    raise RuntimeError(f"GMR reported success but produced no {folder} motion files: {stage_dir}")
        _assert_unchanged(proposal)
        report["outputs"] = {str(path.relative_to(output)): _file(path) for path in sorted(output.rglob("*"))
                             if path.is_file() and path != report_path}
        report["status"] = "CONVERTED_UNREVIEWED"
        save()
        return report
    except BaseException as exc:
        report["status"] = "FAILED_UNREVIEWED"
        report["error"] = str(exc)
        try:
            _assert_unchanged(proposal)
            report["bound_inputs_unchanged"] = True
        except ValueError as input_error:
            report["bound_inputs_unchanged"] = False
            report["input_error"] = str(input_error)
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory outside upstream/input trees")
    parser.add_argument("--timeout", type=float, default=None, help="Optional per-stage conversion timeout in seconds")
    args = parser.parse_args(argv)
    try:
        proposal = plan(args.config, args.output)
        result = proposal if args.action == "plan" else preflight(proposal) if args.action == "preflight" else execute(proposal, args.timeout)
        print(json.dumps(result, indent=2))
        return int(args.action == "preflight" and not result["ok"])
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"GMR adapter: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
