"""Local Blender process execution and evidence binding; no model-provider dependency."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


class BlenderRuntimeError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Nonfinite JSON number")


def load_json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


class BlenderWorker:
    def __init__(self, config):
        self.config = config

    def __call__(self, scene, operations, out):
        out = Path(out)
        request_path, response_path = out.with_suffix(".request.json"), out.with_suffix(".response.json")
        request = {"schema_version": 1, "input_blend_sha256": digest(scene),
                   "output_dir": str(out), "spec": self.config["spec"], "operations": operations}
        save_json(request_path, request)
        binary = shutil.which(self.config["blender"])
        if binary is None:
            raise BlenderRuntimeError("Configured Blender executable was not found")
        worker = Path(__file__).with_name("blender_worker.py")
        env = dict(os.environ)
        for key in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_FILE", "OPENAI_API_KEY"):
            env.pop(key, None)
        for field, folder in (("BLENDER_USER_CONFIG", "config"), ("BLENDER_USER_SCRIPTS", "scripts"),
                              ("BLENDER_USER_EXTENSIONS", "extensions")):
            env[field] = str(out.parent / "blender_profile" / folder)
            Path(env[field]).mkdir(parents=True, exist_ok=True)
        command = [binary, "--background", "--factory-startup", "--disable-autoexec", "--threads", "2", str(scene),
                   "--python-exit-code", "1", "--python", str(worker), "--",
                   "--request", str(request_path), "--response", str(response_path)]
        try:
            result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                    timeout=self.config["worker_timeout"], check=False)
        except subprocess.TimeoutExpired:
            raise BlenderRuntimeError("Blender worker timed out; original checkpoint preserved") from None
        (out.with_suffix(".blender.log")).write_bytes(result.stdout + result.stderr)
        if not response_path.is_file():
            raise BlenderRuntimeError("Blender worker returned no response; inspect its saved log")
        response = load_json(response_path.read_text())
        if result.returncode != 0 and response.get("status") == "ok":
            raise BlenderRuntimeError("Blender exited unsuccessfully despite an ok response")
        return response


def bind_observation(response, out):
    """Only observations physically written by this worker round are sent."""
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise BlenderRuntimeError("Cannot bind an unsuccessful Blender observation")
    response = deepcopy(response)
    root = Path(out).resolve()
    def bind(path):
        p = Path(path).resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file():
            raise BlenderRuntimeError("Worker artifact escaped its output directory")
        return {"path": str(p), "sha256": digest(p)}
    response["checkpoint_binding"] = bind(response["checkpoint"])
    response["native_binding"] = bind(response["native_motion"])
    for index, record in enumerate(response["renders"]):
        record.update(bind(record["path"]))
        record["image_id"] = f"{root.name}/{record['stage']}/{index}/{record['camera']}"
    for record in response.get('inspection_reports', []):
        record.update(bind(record['path']))
    after = [r for r in response["renders"] if r["stage"] == "after"]
    if len({r["camera"] for r in after}) < 2:
        raise BlenderRuntimeError("Worker did not return two after views")
    return response


def verify_binding(record):
    if digest(record["path"]) != record["sha256"]:
        raise BlenderRuntimeError("A checkpoint or evidence file changed after binding")
