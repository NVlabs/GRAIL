"""Optional, local GraspGenX runtime boundary (standard library only).

The host never imports Torch or GraspGenX. Models and source are supplied by the
user and verified by the external worker; nothing is installed or downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

PINNED_COMMIT = "b9429097728cb1c430dd78b92edf17ba318aad03"
PINNED_TREE = "8e82818917d7ca09c9087c777bb7a2783aa016ca"
MODEL_REPOSITORY = "adithyamurali/GraspGenXModel"
MODEL_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
MODEL_FILES = {
    "release/gen/config.yaml": (4006, "16aedbdc23441a1db818261b865f55bd71451d6224fc514615e0d1ab744b7dcb"),
    "release/gen/epoch_736.pth": (1210918342, "8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44"),
    "release/dis/config.yaml": (4024, "43cf8ea0ade67d938040bde0df318b3f4c9c19e4f88d97052dc175df2c28c8e7"),
    "release/dis/epoch_1056.pth": (483889478, "cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime(runtime):
    """Normalize explicit runtime paths without changing the supplied venv."""
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object with python, repo and checkpoint_dir")
    result = {}
    for key in ("python", "repo", "checkpoint_dir"):
        value = runtime.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("runtime." + key + " must be an explicit local path")
        # Do not resolve the executable symlink: doing so loses venv selection.
        path = Path(os.path.abspath(os.path.expanduser(value)))
        if not path.is_absolute() or not path.exists():
            raise ValueError("Missing runtime." + key + ": " + str(path))
        if key == "python":
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError("Runtime Python is not executable: " + str(path))
        elif not path.is_dir():
            raise ValueError("runtime." + key + " must be a directory")
        result[key] = str(path if key == "python" else path.resolve())
    device = runtime.get("device", "cuda:0")
    if not isinstance(device, str) or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
        raise ValueError("runtime.device must name an explicit CUDA device, e.g. cuda:0")
    result["device"] = device
    return result


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=30)
    if proc.returncode:
        raise ValueError("Cannot verify pinned GraspGenX checkout: " + proc.stderr.decode(errors="replace").strip())
    return proc.stdout


def verify_assets(runtime):
    """Check every importable source byte and all four model files before import."""
    runtime = validate_runtime(runtime)
    repo = Path(runtime["repo"])
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    tree = _git(repo, "rev-parse", "HEAD:graspgenx").decode().strip()
    if commit != PINNED_COMMIT or tree != PINNED_TREE:
        raise ValueError("GraspGenX checkout must use pinned commit " + PINNED_COMMIT)
    entries = _git(repo, "ls-tree", "-r", "-z", "HEAD", "--", "graspgenx").split(b"\0")
    tracked, source_hashes = set(), {}
    for entry in entries:
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = raw_name.decode()
        path = repo / name
        if kind != "blob" or mode not in ("100644", "100755") or path.is_symlink() or not path.is_file():
            raise ValueError("Unsupported or missing pinned source: " + name)
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != oid:
            raise ValueError("Pinned GraspGenX source changed: " + name)
        tracked.add(name)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    for path in (repo / "graspgenx").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            name = path.relative_to(repo).as_posix()
            if name not in tracked:
                raise ValueError("Untracked file in pinned GraspGenX package: " + name)
    weights = {}
    for name, (size, expected) in MODEL_FILES.items():
        path = Path(runtime["checkpoint_dir"]) / name
        if not path.is_file():
            raise ValueError("Missing local GraspGenX model file: " + str(path) + "; provision the pinned weights explicitly")
        if path.stat().st_size != size:
            raise ValueError("GraspGenX model size mismatch: " + name)
        digest = file_hash(path)
        if digest != expected:
            raise ValueError("GraspGenX model SHA256 mismatch: " + name)
        weights[name] = dict(size=size, sha256=digest)
    return dict(commit=commit, package_tree=tree, source_sha256=source_hashes,
                model_repository=MODEL_REPOSITORY, model_revision=MODEL_REVISION,
                model_files=weights)


def worker_command(runtime, request_path):
    runtime = validate_runtime(runtime)
    return [runtime["python"], "-I", str(Path(__file__).with_name("graspgen_worker.py")),
            "--request", str(Path(request_path).resolve(strict=True))]


def run_worker(request_path, runtime, *, timeout=None):
    """Run the foreground external worker and return its structured receipt."""
    try:
        proc = subprocess.run(worker_command(runtime, request_path), capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return dict(ok=False, status="RUNTIME_ERROR", selected=False,
                    errors=[str(exc)], checks={})
    try:
        report = json.loads(proc.stdout)
        if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
            raise ValueError("Worker receipt lacks boolean ok")
    except (ValueError, TypeError) as exc:
        return dict(ok=False, status="RUNTIME_ERROR", selected=False, checks={},
                    errors=["External GraspGenX worker returned no valid JSON receipt: " + str(exc)],
                    returncode=proc.returncode, stderr_tail=proc.stderr[-8000:])
    if proc.returncode != 0 and report.get("ok"):
        report.update(ok=False, status="RUNTIME_ERROR", errors=["Worker exited with nonzero status"])
    report["returncode"] = proc.returncode
    if proc.stderr:
        report["stderr_tail"] = proc.stderr[-8000:]
    return report


def preflight(runtime, *, timeout=120.0):
    """Verify external assets, imports and CUDA availability; do not load models."""
    try:
        runtime = validate_runtime(runtime)
    except (OSError, ValueError) as exc:
        return dict(ok=False, status="PREFLIGHT_FAILED", selected=False, checks={}, errors=[str(exc)])
    with tempfile.TemporaryDirectory(prefix="grail_graspgen_preflight_") as folder:
        path = Path(folder) / "request.json"
        path.write_text(json.dumps(dict(action="preflight", runtime=runtime)), encoding="utf-8")
        return run_worker(path, runtime, timeout=timeout)
