"""NumPy-only bindings between live Blender controls and an external repair.

This module never imports Blender, MuJoCo or SciPy. A request snapshots current
native samples; results cannot be consumed against another scene state.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contact_edit import native_io


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_dependencies(path):
    """Bind MJCF includes and file assets without importing a model runtime."""
    path = Path(path).resolve(strict=True)
    pending, paths, roots = [path], set(), []
    while pending:
        current = pending.pop().resolve(strict=True)
        if current in paths:
            continue
        paths.add(current)
        root = ET.parse(current).getroot()
        roots.append(root)
        pending.extend(current.parent / node.attrib["file"] for node in root.iter("include"))
    compiler = {}
    for root in roots:
        for node in root.iter("compiler"):
            compiler.update(node.attrib)
    for root in roots:
        for node in root.iter():
            if node.tag not in {"mesh", "texture", "hfield", "skin"} or "file" not in node.attrib:
                continue
            directory = compiler.get("assetdir", "")
            if node.tag == "mesh":
                directory = compiler.get("meshdir", directory)
            elif node.tag == "texture":
                directory = compiler.get("texturedir", directory)
            name = node.attrib["file"]
            if compiler.get("strippath", "false") == "true":
                name = Path(name).name
            paths.add((path.parent / directory / name).resolve(strict=True))
    return sorted(paths)


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def _native(native, model):
    q = native_io._native_qpos(native)
    joints = native_io._model_joints(model)
    native_io._validate_native_binding(native, joints)
    widths = {0: 7, 1: 4, 2: 1, 3: 1}
    covered = [a for j in joints for a in range(j["address"], j["address"] + widths[j["joint_type"]])]
    if sorted(covered) != list(range(q.shape[1])):
        raise ValueError("Model joint addresses do not cover native qpos")
    for j in joints:
        if j["joint_type"] in (0, 1):
            start = j["address"] + (3 if j["joint_type"] == 0 else 0)
            quats = q[:, start:start + 4]
            if (not np.allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-6, rtol=0)
                    or np.any(np.sum(quats[:-1] * quats[1:], axis=1) < 0)):
                raise ValueError("Native quaternion endpoints/hemispheres are invalid")
        elif j["limited"]:
            native_io.check_sample_limits(model, [j["address"]], q[:, [j["address"]]], tolerance=1e-7)
    return q, joints


def _metadata_equal(parent, candidate, config):
    exceptions = set(native_io.CANDIDATE_STATE_FIELDS) | set(native_io.INVALIDATED_FIELDS)
    exceptions.update(native_io._declared_derived_fields(config))
    exceptions.update({"qpos", "fk_cache_valid"})
    for key, value in parent.items():
        if key in exceptions:
            continue
        if key not in candidate or not _equal(value, candidate[key]):
            raise ValueError("Frozen native metadata changed: " + key)


def _scope(config, parent, model):
    q, joints = _native(parent, model)
    window = config.get("window")
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError("Window needs two source-frame integers")
    start, end = [native_io._integer(v, "source frame") for v in window]
    if not 0 <= start < end < len(q):
        raise ValueError("Window is outside native source frames")
    protected = config.get("protected_frames", [])
    if not isinstance(protected, list):
        raise ValueError("Protected frames must be a list")
    protected = [native_io._integer(v, "protected source frame") for v in protected]
    if len(set(protected)) != len(protected) or any(not start <= v <= end for v in protected):
        raise ValueError("Protected frames must be unique and inside the window")
    frames = [t for t in range(start + 1, end) if t not in protected]
    if not frames:
        raise ValueError("Window has no unprotected interior frames")
    edits = config.get("editable_joints")
    if not isinstance(edits, dict) or not edits:
        raise ValueError("Select at least one editable joint")
    by_name = {j["joint"]: j for j in joints}
    mask = np.zeros(q.shape, dtype=bool)
    records = []
    for name, spec in edits.items():
        if name not in by_name:
            raise ValueError("Unknown native joint: " + name)
        joint = native_io._editable_joint(by_name[name])
        if not isinstance(spec, dict) or set(spec) != {"delta_bounds_rad"}:
            raise ValueError("Joint requires delta_bounds_rad: " + name)
        delta = np.asarray(spec["delta_bounds_rad"])
        if (delta.shape != (2,) or delta.dtype.kind not in "fiu" or not np.isfinite(delta).all()
                or not delta[0] <= 0 <= delta[1] or delta[0] >= delta[1]):
            raise ValueError("Invalid correction corridor: " + name)
        mask[frames, joint["address"]] = True
        records.append((joint, delta.astype(float)))
    return mask, frames, records


def _iteration_seed(config, base, live, model, config_dir):
    """Validate the fixed seed independently of Blender's captured live bytes.

    Native controls store key values as float32. Permit only their numerical
    round trip; a different artist edit must declare a new seed explicitly.
    Frozen samples and metadata retain the exact native export contract.
    """
    iteration = config.get("iteration")
    if iteration is None:
        return None
    if not isinstance(iteration, dict):
        raise ValueError("Repair iteration must be an object")
    path = (config_dir / iteration["seed_npz"]).resolve(strict=True)
    expected = iteration.get("seed_sha256")
    if not isinstance(expected, str) or file_hash(path) != expected:
        raise ValueError("Chosen smooth seed hash mismatch")
    seed = native_io.load_npz(path)
    q, _ = _native(seed, model)
    original = base["qpos"]
    if q.shape != original.shape or q.dtype != original.dtype:
        raise ValueError("Chosen smooth seed shape or dtype differs from original parent")
    _metadata_equal(base, seed, config)
    mask, frames, records = _scope(config, base, model)
    if not np.array_equal(q[~mask], original[~mask]):
        raise ValueError("Chosen smooth seed changed frozen native samples")
    for joint, delta in records:
        difference = q[frames, joint["address"]] - original[frames, joint["address"]]
        if np.any(difference < delta[0] - 1e-10) or np.any(difference > delta[1] + 1e-10):
            raise ValueError("Chosen smooth seed exceeds original correction corridor")
    if not np.array_equal(live["qpos"][~mask], q[~mask]):
        raise ValueError("Live motion changed frozen samples of chosen smooth seed")
    # A delta control's evaluation involves both its reference key and edited
    # key. Bound accumulated float32 error by their magnitudes, never by an
    # arbitrary motion-scale tolerance or the correction corridor.
    rounding = 4 * np.finfo(np.float32).eps * np.maximum(np.abs(q), np.abs(original)) + 2e-12
    if np.any(np.abs(live["qpos"] - q) > rounding):
        raise ValueError("Live motion differs from chosen smooth seed; restore that seed or explicitly revise iteration.seed_npz")
    iteration["seed_npz"] = str(path)
    return path


def prepare_request(config_path, base_native, model_snapshot, live_native, binding, output,
                    *, window=None, joints=None, protected_frames=None, delta_rad=None,
                    samples=None, backend=None, device=None):
    """Snapshot a live sequence into a new job; overrides only narrow its scope."""
    config_path = Path(config_path).resolve(strict=True)
    template = json.loads(config_path.read_text())
    if type(template.get("schema_version")) is not int or template["schema_version"] != 1:
        raise ValueError("Unsupported repair template version")
    config = copy.deepcopy(template)
    if backend is not None:
        if backend not in ("native", "material"):
            raise ValueError("Choose native or material repair backend")
        original_backend = config.get("backend", {"name": "native"})
        if not isinstance(original_backend, dict):
            raise ValueError("Repair backend must be an object")
        config["backend"] = ({"name": "native"} if backend == "native" else
                             {**(original_backend if original_backend.get("name") == "material" else {}),
                              "name": "material"})
    if device is not None:
        import re

        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
            raise ValueError("Repair device must be cpu or an explicit cuda:N device")
        if config.get("backend", {}).get("name") != "material":
            raise ValueError("An explicit repair device requires the material backend")
        config["backend"]["device"] = device
    paths = {key: (config_path.parent / config[key]).resolve(strict=True)
             for key in ("trajectory_npz", "model_xml", "object_mesh_npz")}
    expected = config.get("expected_input_sha256", {})
    if not isinstance(expected, dict) or set(expected) - set(paths):
        raise ValueError("Unknown expected input hash key")
    for key, value in expected.items():
        if file_hash(paths[key]) != value:
            raise ValueError("Original input hash mismatch: " + key)
    original = native_io.load_npz(paths["trajectory_npz"])
    if set(original) != set(base_native) or any(not _equal(value, base_native[key]) for key, value in original.items()):
        raise ValueError("Repair template parent differs from embedded base native motion")
    _native(base_native, model_snapshot)
    _native(live_native, model_snapshot)
    if live_native["qpos"].shape != base_native["qpos"].shape:
        raise ValueError("Live motion frame count differs from embedded base")
    _metadata_equal(base_native, live_native, template)
    _scope(template, base_native, model_snapshot)
    if window is not None:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("Window needs two source-frame integers")
        selected_window = [native_io._integer(v, "source frame") for v in window]
        if not template["window"][0] <= selected_window[0] < selected_window[1] <= template["window"][1]:
            raise ValueError("UI window cannot widen the repair template")
        config["window"] = selected_window
    if joints is not None:
        if (not isinstance(joints, (list, tuple)) or not joints or len(set(joints)) != len(joints)
                or any(name not in template["editable_joints"] for name in joints)):
            raise ValueError("UI joints must be a unique subset of template joints")
        config["editable_joints"] = {name: copy.deepcopy(template["editable_joints"][name]) for name in joints}
    if delta_rad is not None:
        delta_rad = native_io._finite_number(delta_rad, "delta_rad")
        if delta_rad <= 0:
            raise ValueError("delta_rad must be positive")
        template_cap = max(abs(value) for spec in template["editable_joints"].values()
                           for value in spec["delta_bounds_rad"])
        # Blender FloatProperty stores the loaded cap as float32. An unchanged
        # loaded .16 must not shrink to .159999996 and invalidate a seed exactly
        # on its original bound. Restore only that precise UI round trip; an
        # actually smaller user cap still narrows the unchanged source bounds.
        if delta_rad == float(np.float32(template_cap)):
            delta_rad = template_cap
        for spec in config["editable_joints"].values():
            low, high = spec["delta_bounds_rad"]
            spec["delta_bounds_rad"] = [max(low, -delta_rad), min(high, delta_rad)]
    start, end = config["window"]
    protected = [f for f in template.get("protected_frames", []) if start <= f <= end]
    if protected_frames is not None:
        if not isinstance(protected_frames, (list, tuple)):
            raise ValueError("Protected frames must be a list")
        for value in protected_frames:
            value = native_io._integer(value, "protected source frame")
            if not start <= value <= end:
                raise ValueError("Protected frame is outside repair window")
            protected.append(value)
    config["protected_frames"] = sorted(set(protected + [start, end]))
    _scope(config, live_native, model_snapshot)
    seed_path = _iteration_seed(config, base_native, live_native, model_snapshot, config_path.parent)
    if samples is not None:
        if not isinstance(samples, dict) or set(samples) != {"source_frames", "qpos"}:
            raise ValueError("Live samples require source_frames and qpos")
        times = np.asarray(samples["source_frames"])
        poses = np.asarray(samples["qpos"])
        expected_times = audit_times(config, len(live_native["qpos"]))
        if (times.dtype.kind not in "fiu" or not np.array_equal(times, expected_times)
                or poses.dtype.kind != "f" or poses.shape != (len(times), live_native["qpos"].shape[1])
                or not np.isfinite(poses).all()):
            raise ValueError("Live sample coverage must exactly match repair audit frames")
        for index, frame in enumerate(times):
            if float(frame).is_integer() and not np.array_equal(poses[index], live_native["qpos"][int(frame)]):
                raise ValueError("Live integer samples differ from native snapshot")
    dependencies = set(model_dependencies(paths["model_xml"])) | set(paths.values()) | {config_path}
    if seed_path is not None:
        dependencies.add(seed_path)
    for key, path in paths.items():
        config[key] = str(path)
    for env in config.get("environment", []):
        env_path = (config_path.parent / env["mesh_npz"]).resolve(strict=True)
        env["mesh_npz"] = str(env_path)
        dependencies.add(env_path)
    diagnostics = config.get("diagnostics", {})
    for owner, key in ((diagnostics, "baseline_npz"),
                       (diagnostics.get("human_reference", {}), "npz")):
        if key in owner:
            diagnostic_path = (config_path.parent / owner[key]).resolve(strict=True)
            owner[key] = str(diagnostic_path)
            dependencies.add(diagnostic_path)
    input_files = [{"path": str(p), "sha256": file_hash(p)} for p in sorted(dependencies)]
    # Validate serializability before creating a partial job directory.
    scene_binding = copy.deepcopy(binding)
    digest(scene_binding)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_path, model_path = output / "live_native.npz", output / "model_snapshot.npz"
    np.savez_compressed(input_path, **live_native)
    np.savez_compressed(model_path, **model_snapshot)
    if seed_path is None:
        config["trajectory_npz"] = str(input_path)
    config["expected_input_sha256"] = {key: file_hash(config[key]) for key in paths}
    effective = output / "effective_config.json"
    _json(effective, config)
    request = dict(schema_version=1, job_dir=str(output), config_path=str(effective),
                   input_path=str(input_path), model_path=str(model_path), binding=scene_binding,
                   template_path=str(config_path), input_files=input_files,
                   config_sha256=digest(config), config_file_sha256=file_hash(effective),
                   parent_sha256=file_hash(config["trajectory_npz"]), model_sha256=file_hash(model_path))
    if seed_path is not None:
        request.update(parent_path=config["trajectory_npz"], live_sha256=file_hash(input_path),
                       seed_path=str(seed_path), seed_sha256=file_hash(seed_path))
    if samples is not None:
        sample_path = output / "input_samples.npz"
        np.savez_compressed(sample_path, **samples)
        request.update(samples_path=str(sample_path), samples_sha256=file_hash(sample_path))
    _json(output / "request.json", request)
    return request


def validate_request(request, binding=None):
    """Recheck immutable request files and live scene identity before use."""
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported repair request")
    root = Path(request["job_dir"]).resolve(strict=True)
    if json.loads((root / "request.json").read_text()) != request:
        raise ValueError("Repair request changed since capture")
    if binding is not None and digest(binding) != digest(request["binding"]):
        raise ValueError("Stale repair result: live scene binding changed")
    for key, filename, hash_key in (("input_path", "live_native.npz", "live_sha256" if "live_sha256" in request else "parent_sha256"),
                                    ("model_path", "model_snapshot.npz", "model_sha256"),
                                    ("config_path", "effective_config.json", "config_file_sha256")):
        path = Path(request[key]).resolve(strict=True)
        if path != root / filename or file_hash(path) != request[hash_key]:
            raise ValueError("Repair snapshot file changed: " + key)
    for item in request["input_files"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Repair input changed: " + item["path"])
    if "samples_path" in request:
        path = Path(request["samples_path"]).resolve(strict=True)
        if path != root / "input_samples.npz" or file_hash(path) != request["samples_sha256"]:
            raise ValueError("Original live samples changed")
    config = json.loads(Path(request["config_path"]).read_text())
    if digest(config) != request["config_sha256"]:
        raise ValueError("Repair configuration binding changed")
    if "seed_path" in request:
        iteration = config.get("iteration", {})
        if (config["trajectory_npz"] != request["parent_path"]
                or file_hash(request["parent_path"]) != request["parent_sha256"]
                or iteration.get("seed_npz") != request["seed_path"]
                or iteration.get("seed_sha256") != request["seed_sha256"]
                or file_hash(request["seed_path"]) != request["seed_sha256"]):
            raise ValueError("Original parent or chosen smooth seed binding changed")
    return config


def audit_times(config, frame_count):
    """Same finite sampling contract as AuditContext, without its runtimes."""
    audit = config.get("audit", {})
    substeps = native_io._integer(audit.get("substeps", 4), "audit substeps")
    halo = native_io._integer(audit.get("halo", 2), "audit halo")
    if not 1 <= substeps <= 32 or halo < 1:
        raise ValueError("Invalid audit sampling configuration")
    if config.get("iteration") is not None:
        halo = max(3, halo)
    start = max(0, config["window"][0] - halo)
    end = min(frame_count - 1, config["window"][1] + halo)
    source = config.get("source", {})
    guides = [guide["frame"] for guide in source.get("observations", [])
              if start <= guide["frame"] <= end]
    return np.asarray(sorted(set((np.arange(start * substeps, end * substeps + 1) / substeps).tolist() + guides)))


def feasible_float32_key(reference, base, proposed, lower, upper, *, native_dtype=np.float64):
    """Choose a representable Action key whose decoded native pose is feasible.

    Blender evaluates ``base + (key - reference)``. Rounding a solver endpoint
    to float32 can place this decoded pose outside its unchanged corridor. Move
    only to the adjacent representable key toward the feasible interior; never
    loosen the native bounds or silently clamp an invalid solver proposal.
    """
    reference, base, proposed, lower, upper = (
        native_io._finite_number(v, name) for v, name in
        ((reference, "reference"), (base, "base"), (proposed, "proposed"),
         (lower, "lower"), (upper, "upper")))
    if lower > upper or not lower - 1e-10 <= proposed <= upper + 1e-10:
        raise ValueError("Native proposal exceeds the unchanged correction corridor")
    dtype = np.dtype(native_dtype)
    if dtype.kind != "f":
        raise ValueError("Native pose dtype must be floating point")
    target = min(max(proposed, lower), upper)
    key = np.float32(reference + (target - base))
    for _ in range(4):
        if not np.isfinite(key):
            break
        decoded = float(np.asarray(base + (float(key) - reference), dtype=dtype))
        if lower <= decoded <= upper:
            return float(key)
        direction = np.float32(np.inf if decoded < lower else -np.inf)
        following = np.nextafter(key, direction)
        # Crossing both bounds in one representable step means there is no
        # feasible float32 key. Keep the original Action and expose the error.
        following_q = float(np.asarray(base + (float(following) - reference), dtype=dtype))
        if (decoded < lower and following_q > upper) or (decoded > upper and following_q < lower):
            break
        key = following
    raise ValueError("No feasible float32 Blender key inside the unchanged native bounds")


def validate_iteration_report(request, audit, bindings):
    """Require the same fixed seed and guard in solver and actual replay audits."""
    if "seed_sha256" not in request:
        return
    if bindings.get("iteration_seed_npz", {}).get("sha256") != request["seed_sha256"]:
        raise ValueError("Audit does not bind the chosen smooth seed")
    guard = audit.get("smoothness", {})
    if not guard.get("configured") or guard.get("seed_sha256") != request["seed_sha256"]:
        raise ValueError("Audit omitted the chosen-seed smoothness check")
    config = json.loads(Path(request["config_path"]).read_text())
    expected = config["iteration"].get("preserve_smoothness", True)
    if guard.get("preserve_smoothness") is not expected:
        raise ValueError("Audit changed the chosen-seed smoothness policy")


def validate_candidate(request, candidate):
    """Independently verify native invariants; report booleans are not trusted."""
    config = validate_request(request)
    parent = native_io.load_npz(request.get("parent_path", request["input_path"]))
    model = native_io.load_npz(request["model_path"])
    mask, frames, records = _scope(config, parent, model)
    q, _ = _native(candidate, model)
    if q.shape != parent["qpos"].shape or q.dtype != parent["qpos"].dtype:
        raise ValueError("Candidate native shape or dtype differs from parent")
    if not np.array_equal(q[~mask], parent["qpos"][~mask]):
        raise ValueError("Candidate changed frozen native samples")
    for joint, delta in records:
        difference = q[frames, joint["address"]] - parent["qpos"][frames, joint["address"]]
        if np.any(difference < delta[0] - 1e-10) or np.any(difference > delta[1] + 1e-10):
            raise ValueError("Candidate exceeds correction corridor: " + joint["joint"])
    _metadata_equal(parent, candidate, config)
    status = np.asarray(candidate.get("status"))
    selected = np.asarray(candidate.get("selected"))
    if (status.shape != () or str(status.item()) not in {"UNREVIEWED", "REJECTED_BY_AUDIT"}
            or selected.shape != () or selected.dtype.kind != "b" or bool(selected)):
        raise ValueError("Repair candidate may not carry motion approval")
    for key in ("accepted", "reviewed", "selection"):
        if key in candidate:
            raise ValueError("Repair candidate carries obsolete approval: " + key)
    return candidate


def read_result(request, binding):
    """Read a worker result only when its request, files and scene still match."""
    validate_request(request, binding)
    root = Path(request["job_dir"])
    receipt = json.loads((root / "completion.json").read_text())
    if receipt.get("request_sha256") != digest(request) or receipt.get("status") != "COMPLETE":
        raise ValueError("Worker did not complete this repair request")
    result_dir = root / "result"
    candidate_path, report_path = result_dir / "candidate_UNREVIEWED.npz", result_dir / "report.json"
    for path, key in ((candidate_path, "candidate_sha256"), (report_path, "report_sha256")):
        if path.resolve().parent != result_dir.resolve() or file_hash(path) != receipt.get(key):
            raise ValueError("Repair result file changed: " + path.name)
    report = json.loads(report_path.read_text())
    if (report.get("candidate_sha256") != receipt["candidate_sha256"]
            or report.get("config_sha256") != request["config_sha256"]
            or report.get("config_file_sha256") != request["config_file_sha256"]
            or report.get("bindings", {}).get("trajectory_npz", {}).get("sha256") != request["parent_sha256"]):
        raise ValueError("Worker report does not bind this native repair")
    validate_iteration_report(request, report.get("audit_after", {}), report.get("bindings", {}))
    candidate = validate_candidate(request, native_io.load_npz(candidate_path))
    if str(candidate["status"].item()) != report.get("status") or report.get("selected") is not False:
        raise ValueError("Candidate/report status disagree")
    for key, expected in (("local_repair_parent_sha256", request["parent_sha256"]),
                          ("local_repair_config_sha256", request["config_sha256"])):
        if key not in candidate or np.asarray(candidate[key]).shape != () or str(candidate[key].item()) != expected:
            raise ValueError("Candidate provenance differs from request: " + key)
    return candidate, report
