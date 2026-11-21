"""Bound, foreground Blender transport for native Stage 2 observation snapshots."""

import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binding(value):
    if isinstance(value, (str, Path)):
        value = {"path": str(value)}
    path = Path(value["path"]).expanduser().resolve(strict=True)
    actual = digest(path)
    if value.get("sha256", actual) != actual:
        raise ValueError(f"Blender input binding changed: {path}")
    return {"path": str(path), "sha256": actual}


def run_blender(request, output_dir, blender, timeout_s=180):
    """Run an isolated CPU worker; never open, drive or overwrite a live UI scene.

    ``mode=build`` renders supplied native observations and saves/reopens a new
    scene. ``mode=inspect`` exports controls from an existing scene, without
    claiming its cached mesh reflects edits. Inputs and outputs are hash-bound.
    """
    spec = deepcopy(request)
    if spec.get("schema_version") != 1:
        raise ValueError("Blender request schema_version must be 1")
    binary = shutil.which(str(blender))
    if binary is None:
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    for name in (
        "input_blend",
        "parent_motion",
        "candidate_motion",
        "before_geometry",
        "after_geometry",
        "scene",
    ):
        if spec.get(name):
            spec[name] = _binding(spec[name])
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec["output_dir"] = str(out)
    request_path, response_path = out / "request.json", out / "report.json"
    request_path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    env = dict(os.environ)
    for key in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_FILE", "OPENAI_API_KEY"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    for key, folder in (
        ("BLENDER_USER_CONFIG", "config"),
        ("BLENDER_USER_SCRIPTS", "scripts"),
        ("BLENDER_USER_EXTENSIONS", "extensions"),
    ):
        env[key] = str(out / "profile" / folder)
        Path(env[key]).mkdir(parents=True)
    worker = Path(__file__).with_name("blender_worker.py")
    command = [
        binary,
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--threads",
        "2",
        "--python-exit-code",
        "1",
        "--python",
        str(worker),
        "--",
        str(request_path),
    ]
    log = out / "blender.log"
    with log.open("wb") as stream:
        try:
            result = subprocess.run(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Blender timed out; see {log}") from error
    if not response_path.is_file():
        raise RuntimeError(f"Blender returned no report (exit {result.returncode}); see {log}")
    response = json.loads(response_path.read_text())
    if result.returncode or response.get("status") != "ok":
        raise RuntimeError(f"Blender failed: {response.get('error', result.returncode)}; see {log}")
    for binding in response.get("artifacts", {}).values():
        path = Path(binding["path"]).resolve(strict=True)
        if not path.is_relative_to(out) or digest(path) != binding["sha256"]:
            raise RuntimeError("Blender artifact binding failed")
    for record in response.get("renders", []):
        path = Path(record["path"]).resolve(strict=True)
        if not path.is_relative_to(out) or digest(path) != record["sha256"]:
            raise RuntimeError("Blender render binding failed")
    response["log"] = str(log)
    return response
