"""Provider-free, foreground Blender editing with persistent native checkpoints.

The calling agent or a person authors operations. This module never requests a
model response or loads a credential. The shared worker performs native scope
checks, renders evidence, exports, and reopens the resulting ``.blend``.
"""
from copy import deepcopy
import math
from pathlib import Path

from .blender_runtime import (
    BlenderRuntimeError, BlenderWorker, bind_observation, digest, load_json,
    save_json, verify_binding,
)


class BlenderEditError(RuntimeError):
    pass


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise BlenderEditError("Expected a regular input file")
    return {"path": str(path), "sha256": digest(path)}


def read_config(path, input_blend=None):
    """Read the existing native worker contract without inference settings."""
    path = Path(path).expanduser().resolve(strict=True)
    config = load_json(path.read_text(encoding="utf8"))
    allowed = {"schema_version", "input_blend", "blender", "goal", "spec",
               "source_images", "worker_timeout", "max_completion_tokens", "request_timeout"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise BlenderEditError("Unknown configuration fields")
    if (type(config.get("schema_version")) is not int or config["schema_version"] != 1
            or not isinstance(config.get("goal"), str) or not config["goal"].strip()):
        raise BlenderEditError("Configuration needs schema_version=1 and a nonempty goal")
    # Old API-specific fields are accepted only to reuse existing case configs.
    # They neither select a provider nor influence this entirely local execution.
    config.pop("max_completion_tokens", None)
    config.pop("request_timeout", None)
    scene_value = input_blend if input_blend is not None else config.get("input_blend")
    if not isinstance(scene_value, (str, Path)) or not str(scene_value):
        raise BlenderEditError("input_blend must be a checkpoint path")
    scene = Path(scene_value).expanduser()
    if not scene.is_absolute() and input_blend is None:
        scene = path.parent / scene
    scene = scene.resolve(strict=True)
    if not scene.is_file() or scene.suffix.lower() != ".blend":
        raise BlenderEditError("input_blend must be a Blender .blend checkpoint")
    config["input_blend"] = str(scene)
    spec = config.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("editable_joints"), dict) or not spec["editable_joints"]:
        raise BlenderEditError("Declare editable native joints in spec")
    for joint, scope in spec["editable_joints"].items():
        if not isinstance(joint, str) or not joint or not isinstance(scope, dict):
            raise BlenderEditError("Each editable joint needs a name and bounds object")
        for field in ("window", "limits_rad"):
            values = scope.get(field)
            if (not isinstance(values, list) or len(values) != 2
                    or not all(_finite(v) for v in values) or values[0] >= values[1]):
                raise BlenderEditError("Each joint needs increasing window and limits_rad pairs")
        if "max_delta_rad" in scope and (not _finite(scope["max_delta_rad"]) or scope["max_delta_rad"] <= 0):
            raise BlenderEditError("max_delta_rad must be positive and finite")
    cameras = spec.get("cameras")
    if (not isinstance(cameras, list) or not 2 <= len(cameras) <= 8
            or not all(isinstance(c, str) and c for c in cameras) or len(set(cameras)) != len(cameras)):
        raise BlenderEditError("Declare two to eight distinct camera names")
    frames = spec.get("observation_frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 32 or not all(_finite(f) and f >= 0 for f in frames):
        raise BlenderEditError("Declare one to 32 nonnegative source observation frames")
    images = config.get("source_images", [])
    if not isinstance(images, list) or not all(isinstance(p, str) and p for p in images):
        raise BlenderEditError("source_images must be an array of paths")
    config["source_images"] = []
    for item in images:
        image = Path(item).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve(strict=True)
        if not image.is_file() or image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise BlenderEditError("Source images must be PNG or JPEG files")
        config["source_images"].append(str(image))
    config.setdefault("worker_timeout", 180)
    if not _finite(config["worker_timeout"]) or config["worker_timeout"] <= 0:
        raise BlenderEditError("worker_timeout must be positive and finite")
    config.setdefault("blender", "blender")
    if not isinstance(config["blender"], str) or not config["blender"]:
        raise BlenderEditError("blender must name an executable")
    return config


def validate_operations(operations, spec):
    """Preflight declared bounds; the native worker also checks actual curves."""
    if not isinstance(operations, list) or len(operations) > 256:
        raise BlenderEditError("operations must be a JSON array of at most 256 operations")
    seen = set()
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("op") not in ("set_joint_key", "delete_joint_key"):
            raise BlenderEditError("Only set_joint_key and delete_joint_key are supported")
        fields = {"op", "joint", "source_frame"}
        if operation["op"] == "set_joint_key":
            fields.add("value_rad")
        if set(operation) != fields:
            raise BlenderEditError("Unexpected or missing operation fields")
        joint = operation["joint"]
        if not isinstance(joint, str) or joint not in spec["editable_joints"]:
            raise BlenderEditError("Joint is outside the declared editable scope")
        scope, frame = spec["editable_joints"][joint], operation["source_frame"]
        if not _finite(frame) or not scope["window"][0] < frame < scope["window"][1]:
            raise BlenderEditError("Key must be strictly inside its editable source window")
        if (joint, frame) in seen:
            raise BlenderEditError("Duplicate operation on the same joint/frame")
        seen.add((joint, frame))
        if operation["op"] == "set_joint_key":
            value = operation["value_rad"]
            if not _finite(value) or not scope["limits_rad"][0] <= value <= scope["limits_rad"][1]:
                raise BlenderEditError("Key exceeds declared native joint limits")


def runtime_bindings():
    package = Path(__file__).parent
    kit = package / "contact_edit"
    files = [package / name for name in ("__init__.py", "_bootstrap.py", "blender_edit.py", "blender_runtime.py", "blender_worker.py",
                                         "inspection.py", "blender_inspection.py", "static_scene.py")]
    files += [kit / name for name in ("__init__.py", "blender_curves.py", "native_io.py")]
    return [binding(p) for p in files if p.is_file()]


def parent_iteration(checkpoint):
    """Link known local iterations without treating arbitrary paths as reviews."""
    scene = Path(checkpoint["path"])
    manifest = scene.parent.parent.parent / "iteration.json"
    if not manifest.is_file():
        return None
    previous = load_json(manifest.read_text(encoding="utf8"))
    recorded = previous.get("observation", {}).get("checkpoint_binding", {})
    if recorded.get("path") != str(scene):
        return None
    if previous.get("status") != "EXPORTED" or recorded != checkpoint:
        raise BlenderEditError("Previous checkpoint is changed or incomplete; preserve it and save manual edits as a new .blend")
    return binding(manifest)


def run(config_path, output, *, operations_path=None, input_blend=None, worker=None):
    """Export exactly one iteration; a new directory is required every time."""
    config_record = binding(config_path)
    config = read_config(config_record["path"], input_blend=input_blend)
    operations_record = binding(operations_path) if operations_path is not None else None
    operations = load_json(Path(operations_record["path"]).read_text(encoding="utf8")) if operations_record else []
    validate_operations(operations, config["spec"])
    original = binding(config["input_blend"])
    parent = parent_iteration(original)
    state = {"schema_version": 1, "execution": "blender_python", "api_calls": 0,
             "status": "RUNNING", "motion_status": "UNREVIEWED", "selected": False,
             "config": config, "configuration": config_record, "parent_checkpoint": original,
             "parent_iteration": parent, "operations_file": operations_record,
             "operations": deepcopy(operations), "runtime_bindings": runtime_bindings(),
             "source_images": [binding(p) for p in config["source_images"]]}
    fixed = [original, config_record, *state["source_images"], *state["runtime_bindings"]]
    fixed += [p for p in (parent, operations_record) if p is not None]
    for record in fixed:
        verify_binding(record)
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    state_path = out / "iteration.json"
    save_json(state_path, state)
    failure = None
    try:
        execute = worker or BlenderWorker(config)
        response = execute(original["path"], operations, out / "candidate")
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise BlenderEditError("Native worker rejected the iteration: " + str(response.get("error", "invalid response") if isinstance(response, dict) else "invalid response"))
        observation = bind_observation(response, out / "candidate")
        # bind_observation binds evidence, while worker diagnostics establish the
        # native checks. No status in either can confer motion acceptance.
        state["observation"] = observation
        state["status"] = "EXPORTED"
    except (BlenderRuntimeError, BlenderEditError, OSError, ValueError, KeyError, TypeError) as error:
        failure = error
        state["status"] = "FAILED"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
    except KeyboardInterrupt as error:
        failure = error
        state["status"] = "INTERRUPTED"
        state["error"] = {"type": "KeyboardInterrupt", "message": "Interrupted; use a new output directory for another attempt"}
    finally:
        try:
            for record in fixed:
                verify_binding(record)
            observation = state.get("observation", {})
            for record in ([observation[k] for k in ("checkpoint_binding", "native_binding") if k in observation]
                           + observation.get("renders", []) + observation.get("inspection_reports", [])):
                verify_binding(record)
        except (BlenderRuntimeError, OSError) as error:
            failure = error
            state["status"] = "INPUT_CHANGED"
            state["error"] = {"type": type(error).__name__, "message": str(error)}
        save_json(state_path, state)
    if failure is not None:
        raise BlenderEditError(f"{state['error']['message']}; evidence saved in {state_path}") from failure
    return {"iteration": str(state_path), "status": state["status"],
            "checkpoint": state["observation"]["checkpoint"],
            "native_motion": state["observation"]["native_motion"],
            "motion_status": "UNREVIEWED", "selected": False, "api_calls": 0}


def add_cli(sub):
    parser = sub.add_parser("blender-edit", help="Inspect or edit a .blend locally; save a new UNREVIEWED checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New iteration directory; existing outputs are never overwritten")
    parser.add_argument("--operations", type=Path, help="JSON operation array; omit for an inspection-only iteration")
    parser.add_argument("--input-blend", type=Path, help="Override input checkpoint for the next iteration")


def run_cli(args):
    return run(args.config, args.output, operations_path=args.operations, input_blend=args.input_blend)
