"""Explicit local repair jobs, reversible Action previews and native re-audits.

Blender owns and reaps each external worker. File load never starts a job.
The solver implementation is bundled but imports its numerical runtime only in
the selected external Python, keeping the UI and solver environments separate.
"""

import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import runtime_adapter as adapter
from .inspection import blender_collision as collision
from .inspection import blender_inspection as inspection
from .repair_runtime import repair_bridge as bridge

_states = {}
_registered = False
_INTERVAL = 0.2


def _path(raw, label):
    if not raw.strip():
        raise ValueError("Choose " + label + ".")
    return Path(bpy.path.abspath(raw)).expanduser().resolve()


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _action_record(action):
    if action is None:
        return None
    return dict(
        name=action.name,
        curves=[
            dict(
                path=fc.data_path,
                index=fc.array_index,
                mute=fc.mute,
                extrapolation=fc.extrapolation,
                smoothing=fc.auto_smoothing,
                modifiers=[m.type for m in fc.modifiers],
                keys=[
                    dict(
                        co=list(k.co),
                        left=list(k.handle_left),
                        right=list(k.handle_right),
                        left_type=k.handle_left_type,
                        right_type=k.handle_right_type,
                        interpolation=k.interpolation,
                        easing=k.easing,
                        amplitude=k.amplitude,
                        back=k.back,
                        period=k.period,
                    )
                    for k in fc.keyframe_points
                ],
            )
            for fc in action.fcurves
        ],
    )


def _binding(scene):
    """Bind whole Actions and original mesh data, independent of playhead time."""
    s = scene.grail_repair
    texts = {
        name: hashlib.sha256(bpy.data.texts[name].as_string().encode()).hexdigest()
        for name in adapter.TEXT_NAMES
        + (
            "NativeModel.npz.b64",
            "PortableReference.npz.b64",
            "Enable_Exact_Native_Preview.py",
        )
        if name in bpy.data.texts
    }
    actions = {}
    for name in (adapter.runtime.CONTROLLER, "Native_Qpos_Controller"):
        obj = scene.objects.get(name)
        if obj:
            anim = obj.animation_data
            if anim and (anim.drivers or anim.nla_tracks):
                raise ValueError(
                    "Repair supports explicit native Actions; bake/review drivers or NLA first."
                )
            actions[name] = _action_record(anim.action if anim else None)
    for name in (adapter.runtime.REFERENCE, "Native_Qpos_Reference"):
        actions[name] = _action_record(bpy.data.actions.get(name))
    meshes = {}
    for obj in scene.objects:
        if (
            obj.type != "MESH"
            or obj.get("grail_collision_owner")
            or obj.get("grail_inspection_owner")
        ):
            continue
        vertices = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", vertices)
        loops = np.empty(len(obj.data.loops), dtype=np.int32)
        obj.data.loops.foreach_get("vertex_index", loops)
        sizes = np.empty(len(obj.data.polygons), dtype=np.int32)
        obj.data.polygons.foreach_get("loop_total", sizes)
        meshes[obj.name] = dict(
            data=hashlib.sha256(
                vertices.tobytes() + loops.tobytes() + sizes.tobytes()
            ).hexdigest(),
            parent=obj.parent.name if obj.parent else None,
            parent_inverse=[list(r) for r in obj.matrix_parent_inverse],
            scale=list(obj.scale),
            # FK updates mounted body transforms, while original visual mesh
            # offsets relative to those bodies must remain fixed.
            mounted_basis=(
                [list(r) for r in obj.matrix_basis]
                if obj.parent and obj.parent.name.startswith("G1_")
                else None
            ),
            modifiers=[m.type for m in obj.modifiers],
            constraints=[c.type for c in obj.constraints],
        )
    config = _path(s.config_path, "a repair config")
    return _json_digest(
        dict(
            texts=texts,
            actions=actions,
            meshes=meshes,
            timing=[
                scene.frame_start,
                scene.frame_end,
                scene.render.fps,
                scene.render.fps_base,
            ],
            settings=dict(
                config=str(config),
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                python=str(_path(s.python_path, "an external Python runtime")),
                window=[s.window_start, s.window_end],
                protected=s.protected_frames,
                delta=s.delta_rad,
                joints=[(j.name, j.enabled) for j in s.joints],
                backend=s.backend,
                device=s.device if s.backend == "material" else "cpu",
            ),
        )
    )


def _set_status(scene, status, message):
    scene.grail_repair.status = status
    scene.grail_repair.message = message
    for window in bpy.context.window_manager.windows:
        if window.scene == scene:
            for area in window.screen.areas:
                area.tag_redraw()


def _clear_preview_record(scene):
    if "local_repair_preview_record" in scene:
        del scene["local_repair_preview_record"]


def update_scene(scene):
    adapter.upgrade_compatible_scene()
    _set_status(
        scene,
        "IDLE",
        "Compatible exporter binding updated. Native motion and curves are preserved.",
    )


def _active(scene):
    if scene != bpy.context.scene:
        raise ValueError("Switch to the scene that owns these native controls.")
    state = adapter.initialize()
    if "Native_Qpos_Controller" in scene.objects and state.get("preview") is None:
        raise ValueError(
            "Enable Exact Preview for this trusted robot scene before local repair."
        )
    adapter.runtime.validate_curves()
    adapter.validate_protected_keys()
    return state


def _times(config, count):
    return bridge.audit_times(config, count)


def _sample(times):
    return np.array([adapter.runtime.q_at(float(t)) for t in times])


def _native_snapshot(state):
    sampled = _sample(range(len(state["native"]["qpos"])))
    native, _ = adapter.native_io.merge_curve_samples(
        state["native"], state["model"], state["plan"], sampled
    )
    return native


def load_setup(scene):
    if _states.get(scene.as_pointer(), {}).get("process"):
        raise ValueError("Cancel the running repair before changing setup.")
    s = scene.grail_repair
    config_path = _path(s.config_path, "a repair configuration JSON")
    config = json.loads(config_path.read_text())
    iteration = config.get("iteration")
    s.iteration_seed = ""
    s.smoothness_status = "NOT_CONFIGURED"
    if iteration is not None:
        if not isinstance(iteration, dict) or not isinstance(iteration.get("seed_npz"), str):
            raise ValueError("Iteration setup needs the chosen smooth seed NPZ.")
        seed = (config_path.parent / iteration["seed_npz"]).resolve(strict=True)
        if bridge.file_hash(seed) != iteration.get("seed_sha256"):
            raise ValueError("Chosen smooth seed hash mismatch. Restore the bound file or revise the setup explicitly.")
        s.iteration_seed = str(seed)
        s.smoothness_status = "PENDING"
    backend = config.get("backend", {"name": "native"})
    if not isinstance(backend, dict) or backend.get("name") not in ("native", "material"):
        raise ValueError("Choose native or material repair backend in the setup.")
    s.backend = backend["name"]
    s.device = backend.get("device", "cpu" if s.backend == "material" else "cuda:0")
    s.window_start, s.window_end = config["window"]
    s.protected_frames = ", ".join(str(f) for f in config.get("protected_frames", []))
    s.joints.clear()
    for name in config["editable_joints"]:
        item = s.joints.add()
        item.name = name
        item.enabled = True
    s.delta_rad = max(
        abs(v)
        for item in config["editable_joints"].values()
        for v in item["delta_bounds_rad"]
    )
    _set_status(
        scene,
        "IDLE",
        ("Chosen smooth seed loaded. Original parent, bounds and contacts stay fixed; live curves must match this seed."
         if iteration is not None else
         "Scope loaded. Fixed contacts and original geometry come from this configuration."),
    )


def use_timeline(scene):
    s = scene.grail_repair
    s.window_start = (
        scene.frame_preview_start if scene.use_preview_range else scene.frame_start
    ) - 1
    s.window_end = (
        scene.frame_preview_end if scene.use_preview_range else scene.frame_end
    ) - 1


def _journal(state, status, **extra):
    path = Path(state["request"]["job_dir"]) / "ui_lifecycle.json"
    path.write_text(
        json.dumps(
            dict(
                status=status,
                kind=state.get("kind"),
                pid=state["process"].pid if state.get("process") else None,
                time=datetime.now(timezone.utc).isoformat(),
                **extra,
            ),
            indent=2,
        )
        + "\n"
    )


def _launch(scene, state, kind, *, candidate=None, samples=None, output=None):
    s = scene.grail_repair
    python = _path(s.python_path, "an external Python with NumPy, SciPy and MuJoCo")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("The selected external Python is not executable.")
    worker = Path(bridge.__file__).with_name("repair_worker.py")
    command = [
        str(python),
        str(worker),
        "--request",
        str(Path(state["request"]["job_dir"]) / "request.json"),
    ]
    if kind == "preview_audit":
        command += [
            "--audit-candidate",
            str(candidate),
            "--audit-samples",
            str(samples),
            "--output",
            str(output),
        ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    folder = Path(state["request"]["job_dir"])
    logtag = kind + "_" + uuid.uuid4().hex[:8]
    streams = [
        (folder / (logtag + ".stdout.log")).open("w"),
        (folder / (logtag + ".stderr.log")).open("w"),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=folder,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=streams[0],
            stderr=streams[1],
        )
    except Exception:
        for stream in streams:
            stream.close()
        raise
    state.update(
        process=process,
        streams=streams,
        started=time.monotonic(),
        kind=kind,
        audit_output=str(output) if output else None,
    )
    _journal(state, "RUNNING")
    _set_status(
        scene,
        "PREVIEW_CHECKING" if kind == "preview_audit" else "RUNNING",
        "Checking the actual Blender preview…"
        if kind == "preview_audit"
        else "Repairing current native motion in the external worker…",
    )
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=_INTERVAL)


def start(scene):
    pointer = scene.as_pointer()
    previous = _states.get(pointer)
    if previous and (previous.get("process") or previous.get("preview_action")):
        raise ValueError("Finish or restore the current preview before another repair.")
    state = _active(scene)
    s = scene.grail_repair
    names = [j.name for j in s.joints if j.enabled]
    if not names:
        raise ValueError("Load a repair setup and select its permitted joints.")
    channels = {c["joint"]: c for c in state["channels"]}
    controller = bpy.data.objects[adapter.runtime.CONTROLLER]
    action = controller.animation_data.action
    protected = [
        int(value.strip()) for value in s.protected_frames.split(",") if value.strip()
    ]
    for name in names:
        if name not in channels:
            raise ValueError(
                "Create native controls for this repair joint first: " + name
            )
        c = channels[name]
        if not c["window"][0] <= s.window_start < s.window_end <= c["window"][1]:
            raise ValueError(
                "Repair window must stay inside the authored control scope: " + name
            )
        fc = action.fcurves.find(f'["qpos_{c["address"]:02d}"]', index=0)
        if any(k.interpolation != "LINEAR" for k in fc.keyframe_points):
            raise ValueError(
                "Repair Preview currently requires LINEAR native curves. Export/rebase authored Bezier motion first."
            )
        key_times = [float(k.co[0] - 1) for k in fc.keyframe_points]
        if s.window_start not in key_times or s.window_end not in key_times:
            raise ValueError(
                "Add native keys at both repair-window boundaries before running repair."
            )
        plan_channel = next(p for p in state["plan"]["channels"] if p["joint"] == name)
        for frame in plan_channel.get("protected_source_frames", []):
            if s.window_start <= frame <= s.window_end:
                if int(frame) != frame:
                    raise ValueError(
                        "This repair supports protected integer-frame milestones only."
                    )
                protected.append(int(frame))
    snapshot = _native_snapshot(state)
    template_path = _path(s.config_path, "a repair configuration JSON")
    effective = json.loads(template_path.read_text())
    effective["window"] = [s.window_start, s.window_end]
    times = _times(effective, len(snapshot["qpos"]))
    sampled = _sample(times)
    lo = np.floor(times).astype(int)
    hi = np.minimum(lo + 1, len(snapshot["qpos"]) - 1)
    alpha = (times - lo)[:, None]
    linear = snapshot["qpos"][lo] * (1 - alpha) + snapshot["qpos"][hi] * alpha
    if np.max(np.abs(sampled - linear)) > 1e-6:
        raise ValueError(
            "Current curves differ from native linear replay in the audit window. Export/rebase them before local repair."
        )
    root = _path(s.output_root, "a repair output folder")
    root.mkdir(parents=True, exist_ok=True)
    folder = root / (
        "repair_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    binding = _binding(scene)
    request = bridge.prepare_request(
        template_path,
        state["native"],
        state["model"],
        snapshot,
        binding,
        folder,
        window=[s.window_start, s.window_end],
        joints=names,
        protected_frames=sorted(set(protected)),
        delta_rad=s.delta_rad,
        samples=dict(source_frames=times, qpos=sampled),
        backend=s.backend,
        device=s.device if s.backend == "material" else None,
    )
    job = dict(
        request=request,
        binding=binding,
        original_action=action,
        process=None,
        preview_action=None,
        candidate=None,
        report=None,
        parent_samples=sampled,
        sample_times=times,
    )
    _states[pointer] = job
    s.last_output = str(folder)
    _launch(scene, job, "repair")
    return request


def _close(state, terminate=False):
    process = state.get("process")
    if process:
        if terminate and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        elif process.poll() is not None:
            process.wait()
    for stream in state.get("streams", []):
        stream.close()
    state["streams"] = []
    state["process"] = None


def _refresh(scene):
    if scene == bpy.context.scene:
        adapter.runtime.contact_edit_preview(scene)
        bpy.context.view_layer.update()
    collision.mark_collision_stale(scene)
    inspection.mark_dashboard_stale(scene)
    # Existing automatic Newton and dashboard handlers observe the changed
    # native Action. No old contact cache is marked valid for the new pose.
    from . import collision_live

    live = collision_live._states.get(scene.as_pointer())
    if live:
        collision_live._touch(scene, live)


def _restore_action(scene, state):
    action = state.get("preview_action")
    if action:
        controller = scene.objects.get(adapter.runtime.CONTROLLER)
        if (
            controller
            and controller.animation_data
            and controller.animation_data.action == action
        ):
            controller.animation_data.action = state["original_action"]
        state["preview_action"] = None
        _clear_preview_record(scene)
        # Keep diagnostic edits/checkpoints instead of destroying user work.
        action.use_fake_user = True
        _refresh(scene)


def cancel(scene):
    state = _states.get(scene.as_pointer())
    if state:
        _close(state, terminate=True)
        _restore_action(scene, state)
        _journal(state, "CANCELLED")
    _set_status(
        scene,
        "CANCELLED",
        "Worker stopped. Original action retained; logs remain in the output folder.",
    )


def _finish(scene, state):
    process = state["process"]
    if process.poll() is None:
        if time.monotonic() - state["started"] > scene.grail_repair.timeout_s:
            cancel(scene)
            raise RuntimeError("Repair worker timed out; see its saved logs.")
        return
    code = process.returncode
    kind = state["kind"]
    _close(state)
    expected = (
        state.get("preview_binding") if kind == "preview_audit" else state["binding"]
    )
    if _binding(scene) != expected:
        _journal(state, "STALE")
        _set_status(
            scene,
            "STALE",
            "Animation, geometry or repair setup changed. Result is retained as outdated; run again.",
        )
        return
    if code not in (0, 1):
        raise RuntimeError(
            f"Repair worker failed ({code}); logs: {state['request']['job_dir']}"
        )
    if kind == "repair":
        folder = Path(state["request"]["job_dir"])
        if not (folder / "completion.json").is_file():
            error_path = folder / "worker_error.json"
            if error_path.is_file():
                error = json.loads(error_path.read_text())
                raise RuntimeError(f"{error.get('type', 'Worker error')}: {error.get('message', '')}")
            raise RuntimeError(f"Repair worker produced no completion receipt; see logs: {folder}")
        candidate, report = bridge.read_result(state["request"], state["binding"])
        state.update(candidate=candidate, report=report)
        scene.grail_repair.smoothness_status = _smoothness_status(report["audit_after"])
        scope = report["audit_after"]["affected_scope_passed"]
        _set_status(
            scene,
            "READY",
            "Candidate ready for Preview. "
            + ("Local checks passed. " if scope else "Local checks still fail. ")
            + (
                "Full audit passed; motion UNREVIEWED."
                if report["audit_after"]["passed"]
                else "Full audit remains REJECTED; inspect failures."
            ) + _smoothness_message(report["audit_after"]),
        )
    else:
        path = Path(state["audit_output"]) / "report.json"
        if not path.is_file():
            error_path = Path(state["request"]["job_dir"]) / "worker_error.json"
            if error_path.is_file():
                error = json.loads(error_path.read_text())
                raise RuntimeError(f"Preview audit {error.get('type', 'worker error')}: {error.get('message', '')}")
            raise RuntimeError(f"Preview audit produced no report; see logs: {state['request']['job_dir']}")
        report = json.loads(path.read_text())
        bridge.validate_request(state["request"])
        if (
            report.get("request_sha256") != bridge.digest(state["request"])
            or report.get("config_sha256") != state["request"]["config_sha256"]
        ):
            raise ValueError("Preview audit belongs to another repair request.")
        if (
            report.get("candidate_sha256")
            != hashlib.sha256(
                Path(state["preview_candidate_path"]).read_bytes()
            ).hexdigest()
        ):
            raise ValueError(
                "Preview audit is not bound to the displayed native candidate."
            )
        if not report.get("supplied_poses_audited"):
            raise ValueError("Actual Blender subframes were not checked by the worker.")
        bridge.validate_iteration_report(state["request"], report.get("audit_after", {}), report.get("bindings", {}))
        if (
            report.get("samples_sha256")
            != hashlib.sha256(
                Path(state["preview_samples_path"]).read_bytes()
            ).hexdigest()
        ):
            raise ValueError("Preview samples changed after audit.")
        state["preview_report"] = report
        scene.grail_repair.smoothness_status = _smoothness_status(report["audit_after"])
        state["preview_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _set_status(
            scene,
            "PREVIEW_READY",
            "Actual Blender preview checked. "
            + (
                "Declared audit passed; motion UNREVIEWED."
                if report["audit_after"]["passed"]
                else "Audit failures remain; inspect the saved checks."
            ) + _smoothness_message(report["audit_after"]),
        )
    _journal(state, scene.grail_repair.status)


def _smoothness_status(audit):
    return audit.get("checks", {}).get("smoothness", {}).get("status", "NOT_CONFIGURED")


def _smoothness_message(audit):
    guard = audit.get("smoothness", {})
    if not guard.get("configured"):
        return ""
    if guard.get("passed"):
        return " Chosen-seed smoothness checks passed."
    if guard.get("preserve_smoothness"):
        return " Smoothness regressed; Restore the chosen seed. Apply is blocked."
    return " Smoothness measured; preservation policy is disabled."


def _tick():
    if not _registered:
        return None
    scenes = {scene.as_pointer(): scene for scene in bpy.data.scenes}
    active = False
    for pointer, state in list(_states.items()):
        if not state.get("process"):
            continue
        scene = scenes.get(pointer)
        if scene is None:
            _close(state, terminate=True)
            _states.pop(pointer, None)
            continue
        try:
            _finish(scene, state)
        except Exception as exc:
            _close(state, terminate=True)
            _journal(state, "FAILED", error=str(exc))
            _set_status(scene, "FAILED", str(exc))
        active |= state.get("process") is not None
    return _INTERVAL if active else None


def preview(scene):
    state = _states.get(scene.as_pointer())
    if (
        not state
        or state.get("process")
        or state.get("preview_action")
        or state.get("candidate") is None
    ):
        raise ValueError("Run repair and wait for a ready candidate first.")
    current = _active(scene)
    candidate, _report = bridge.read_result(state["request"], _binding(scene))
    controller = bpy.data.objects[adapter.runtime.CONTROLLER]
    state["original_action"].use_fake_user = True
    action = controller.animation_data.action.copy()
    action.name = "LocalRepair_Preview_UNREVIEWED"
    action.use_fake_user = True
    state["preview_action"] = action
    config = json.loads(Path(state["request"]["config_path"]).read_text())
    start, end = config["window"]
    protected = set(config.get("protected_frames", [])) | {start, end}
    reference = bpy.data.actions[adapter.runtime.REFERENCE]
    by_name = {c["joint"]: c for c in current["channels"]}
    native_joints = {j["joint"]: j for j in adapter.native_io._model_joints(current["model"])}
    bound_parent = adapter.native_io.load_npz(state["request"].get("parent_path", state["request"]["input_path"]))
    controller.animation_data.action = action
    try:
        for name in config["editable_joints"]:
            address = by_name[name]["address"]
            path = f'["qpos_{address:02d}"]'
            fc = action.fcurves.find(path, index=0)
            ref = reference.fcurves.find(path, index=0)
            for frame in range(start + 1, end):
                if frame in protected:
                    continue
                for key in list(fc.keyframe_points):
                    if frame == float(key.co[0] - 1):
                        fc.keyframe_points.remove(key, fast=True)
                low, high = config["editable_joints"][name]["delta_bounds_rad"]
                low += float(bound_parent["qpos"][frame, address])
                high += float(bound_parent["qpos"][frame, address])
                joint = native_joints[name]
                if joint["limited"]:
                    low = max(low, joint["limits"][0])
                    high = min(high, joint["limits"][1])
                value = bridge.feasible_float32_key(
                    ref.evaluate(frame + 1), float(current["native"]["qpos"][frame, address]),
                    float(candidate["qpos"][frame, address]), low, high,
                    native_dtype=current["native"]["qpos"].dtype)
                key = fc.keyframe_points.insert(frame + 1, value, options={"FAST"})
                key.interpolation = "LINEAR"
            fc.update()
        adapter.runtime.validate_curves()
        adapter.validate_protected_keys()
        actual = _native_snapshot(current)
        allowed = np.zeros(actual["qpos"].shape, dtype=bool)
        frames = [f for f in range(start + 1, end) if f not in protected]
        addresses = [by_name[name]["address"] for name in config["editable_joints"]]
        allowed[np.ix_(frames, addresses)] = True
        parent = adapter.native_io.load_npz(state["request"]["input_path"])
        if not np.array_equal(actual["qpos"][~allowed], parent["qpos"][~allowed]):
            raise ValueError(
                "Preview altered protected or undeclared native coordinates."
            )
        if np.max(np.abs(actual["qpos"] - candidate["qpos"])) > 1e-6:
            raise ValueError(
                "Blender preview does not reproduce the repaired native candidate."
            )
        folder = Path(state["request"]["job_dir"]) / ("preview_" + uuid.uuid4().hex[:8])
        folder.mkdir()
        native_path = folder / "candidate_UNREVIEWED.npz"
        sample_path = folder / "actual_samples.npz"
        np.savez_compressed(native_path, **actual)
        np.savez_compressed(
            sample_path,
            source_frames=state["sample_times"],
            qpos=_sample(state["sample_times"]),
        )
        state.update(
            preview_candidate_path=str(native_path),
            preview_samples_path=str(sample_path),
            preview_binding=_binding(scene),
        )
        scene["local_repair_preview_record"] = json.dumps(
            dict(
                original_action=state["original_action"].name,
                preview_action=action.name,
                original_digest=_json_digest(_action_record(state["original_action"])),
                preview_digest=_json_digest(_action_record(action)),
            )
        )
        _refresh(scene)
        _launch(
            scene,
            state,
            "preview_audit",
            candidate=native_path,
            samples=sample_path,
            output=folder / "audit",
        )
    except Exception:
        _restore_action(scene, state)
        raise


def restore(scene):
    state = _states.get(scene.as_pointer())
    if not state or not state.get("preview_action"):
        raise ValueError("No repair preview to restore.")
    _close(state, terminate=True)
    _restore_action(scene, state)
    current = _binding(scene) == state["binding"]
    _set_status(
        scene,
        "READY" if current else "STALE",
        "Original action restored. Candidate and diagnostics remain saved.",
    )


def apply(scene):
    state = _states.get(scene.as_pointer())
    if (
        not state
        or scene.grail_repair.status != "PREVIEW_READY"
        or state.get("process")
    ):
        raise ValueError(
            "Preview and wait for its actual Blender audit before applying."
        )
    _active(scene)
    bridge.validate_request(state["request"])
    if _binding(scene) != state["preview_binding"]:
        raise ValueError(
            "Preview changed after its audit; restore and run a new repair."
        )
    path = Path(state["audit_output"]) / "report.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != state["preview_report_sha256"]:
        raise ValueError("Preview report changed after it was verified.")
    bridge.validate_iteration_report(state["request"], state["preview_report"]["audit_after"],
                                     state["preview_report"].get("bindings", {}))
    guard = state["preview_report"]["audit_after"].get("smoothness", {})
    if guard.get("preserve_smoothness") and not guard.get("passed"):
        raise ValueError("Smoothness regressed relative to the chosen seed. Restore the seed; this trial cannot be applied.")
    action = state["preview_action"]
    state["original_action"].use_fake_user = True
    action.name = "LocalRepair_Applied_UNREVIEWED"
    state["preview_action"] = None
    _clear_preview_record(scene)
    record = dict(
        schema_version=1,
        status="UNREVIEWED",
        selected=False,
        solver_report=str(Path(state["request"]["job_dir"]) / "result/report.json"),
        displayed_audit=str(path),
        displayed_audit_sha256=state["preview_report_sha256"],
        audit_passed=state["preview_report"]["audit_after"]["passed"],
        smoothness=guard,
        original_action=state["original_action"].name,
        applied_action=action.name,
        note="Explicit Apply changes editable curves; it does not accept the motion.",
    )
    name = "LocalRepair_" + uuid.uuid4().hex[:8] + ".json"
    adapter.runtime.text(name, json.dumps(record, indent=2))
    scene["local_repair_last_record"] = name
    scene["contact_edit_status"] = (
        "UNREVIEWED; local repair applied with retained audit evidence"
    )
    _refresh(scene)
    _set_status(
        scene,
        "APPLIED",
        "Applied as editable native curves, UNREVIEWED. Save/export a new checkpoint.",
    )
    _journal(state, "APPLIED", record=name)


class GRAILRepairJoint(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Edit", default=False)


class GRAILRepairSettings(bpy.types.PropertyGroup):
    show_joints: BoolProperty(name="Joint selection", default=False)
    config_path: StringProperty(name="Repair setup JSON", subtype="FILE_PATH")
    python_path: StringProperty(
        name="Solver Python",
        subtype="FILE_PATH",
        description="External Python with NumPy, SciPy and MuJoCo; material repair also requires Torch and Warp",
    )
    backend: EnumProperty(
        name="Repair method",
        items=(("native", "Native CPU", "Coupled native repair with audited clearance witnesses"),
               ("material", "Material GPU (experimental)", "Fixed contact anchors with differentiable mesh-distance proposals; exact geometry audit remains required")),
        default="native",
    )
    device: StringProperty(name="Device", default="cuda:0",
                           description="Explicit cuda:N device, or cpu for reference validation")
    output_root: StringProperty(name="Repair output folder", subtype="DIR_PATH")
    window_start: IntProperty(name="Start (source)", min=0)
    window_end: IntProperty(name="End (source)", min=1, default=10)
    protected_frames: StringProperty(
        name="Protected frames",
        description="Comma-separated integer source frames; window endpoints are always protected",
    )
    delta_rad: FloatProperty(
        name="Max correction (rad)", default=0.16, min=0.000001, max=6.28, precision=4
    )
    timeout_s: IntProperty(name="Worker timeout (s)", default=600, min=1, max=86400)
    joints: CollectionProperty(type=GRAILRepairJoint)
    status: StringProperty(default="IDLE", options={"SKIP_SAVE"})
    message: StringProperty(
        default="Load a repair setup after native controls are ready.",
        options={"SKIP_SAVE"},
    )
    last_output: StringProperty(name="Last repair output", subtype="DIR_PATH")
    iteration_seed: StringProperty(name="Chosen smooth seed", subtype="FILE_PATH")
    smoothness_status: StringProperty(default="NOT_CONFIGURED", options={"SKIP_SAVE"})


def _execute(operator, context, fn):
    try:
        fn(context.scene)
        return {"FINISHED"}
    except Exception as exc:
        context.scene.grail_repair.message = str(exc)
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}


class GRAILRepair_OT_load(bpy.types.Operator):
    bl_idname = "grail_g1.repair_load_setup"
    bl_label = "Load Repair Setup"

    def execute(self, context):
        return _execute(self, context, load_setup)


class GRAILRepair_OT_update_scene(bpy.types.Operator):
    bl_idname = "grail_g1.repair_update_scene"
    bl_label = "Update Compatible Scene Bindings"
    bl_description = "Update a known metadata-only exporter change in the current scene; does not execute embedded scripts or edit motion"

    def execute(self, context):
        return _execute(self, context, update_scene)


class GRAILRepair_OT_range(bpy.types.Operator):
    bl_idname = "grail_g1.repair_use_timeline"
    bl_label = "Use Timeline Preview Range"

    def execute(self, context):
        return _execute(self, context, use_timeline)


class GRAILRepair_OT_run(bpy.types.Operator):
    bl_idname = "grail_g1.repair_run"
    bl_label = "Run Local Repair"

    def execute(self, context):
        return _execute(self, context, start)


class GRAILRepair_OT_preview(bpy.types.Operator):
    bl_idname = "grail_g1.repair_preview"
    bl_label = "Preview Candidate"

    def execute(self, context):
        return _execute(self, context, preview)


class GRAILRepair_OT_restore(bpy.types.Operator):
    bl_idname = "grail_g1.repair_restore"
    bl_label = "Restore Original"

    def execute(self, context):
        return _execute(self, context, restore)


class GRAILRepair_OT_apply(bpy.types.Operator):
    bl_idname = "grail_g1.repair_apply"
    bl_label = "Apply to Editable Curves"

    def execute(self, context):
        return _execute(self, context, apply)


class GRAILRepair_OT_cancel(bpy.types.Operator):
    bl_idname = "grail_g1.repair_cancel"
    bl_label = "Cancel Worker"

    def execute(self, context):
        return _execute(self, context, cancel)


class GRAILREPAIR_PT_panel(bpy.types.Panel):
    bl_label = "Local Window Repair"
    bl_idname = "GRAILREPAIR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GRAIL G1"
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        s = context.scene.grail_repair
        layout.label(text="Source 0 = Blender 1. Endpoints stay fixed.")
        block = bpy.data.texts.get("ContactEdit_native_io.py")
        if block and block.as_string() != Path(adapter.native_io.__file__).read_text():
            layout.operator("grail_g1.repair_update_scene", icon="FILE_REFRESH")
        layout.prop(s, "config_path")
        layout.operator("grail_g1.repair_load_setup")
        if s.iteration_seed:
            box = layout.box()
            box.label(text="Iterate from chosen smooth seed")
            box.label(text=Path(s.iteration_seed).name)
            box.label(text="Original parent / bounds / contacts retained")
            box.label(text="Smoothness: " + s.smoothness_status)
        row = layout.row(align=True)
        row.prop(s, "window_start")
        row.prop(s, "window_end")
        layout.operator("grail_g1.repair_use_timeline")
        layout.prop(s, "protected_frames")
        layout.prop(s, "delta_rad")
        if s.joints:
            box = layout.box()
            box.prop(
                s,
                "show_joints",
                text=f"Joint selection: {sum(j.enabled for j in s.joints)}/{len(s.joints)}",
            )
            if s.show_joints:
                for item in s.joints:
                    box.prop(item, "enabled", text=item.name)
        layout.prop(s, "python_path")
        layout.prop(s, "backend")
        if s.backend == "material":
            layout.prop(s, "device")
            layout.label(text="External runtime needs Torch + Warp.")
        layout.prop(s, "output_root")
        row = layout.row()
        row.enabled = s.status not in ("RUNNING", "PREVIEW_CHECKING")
        row.operator("grail_g1.repair_run", icon="PLAY")
        row = layout.row(align=True)
        row.operator("grail_g1.repair_preview")
        row.operator("grail_g1.repair_restore")
        row = layout.row()
        row.enabled = s.status == "PREVIEW_READY"
        row.operator("grail_g1.repair_apply", icon="CHECKMARK")
        if s.status in ("RUNNING", "PREVIEW_CHECKING"):
            layout.operator("grail_g1.repair_cancel", icon="CANCEL")
        layout.label(text="Apply remains UNREVIEWED.")
        box = layout.box()
        box.label(text=s.status)
        import textwrap

        for line in textwrap.wrap(s.message, width=46):
            box.label(text=line)


@persistent
def _load_pre(_unused):
    for scene in bpy.data.scenes:
        state = _states.get(scene.as_pointer())
        if state:
            _close(state, terminate=True)
            _restore_action(scene, state)
    _states.clear()


@persistent
def _load_post(_unused):
    """Restore a saved transient preview without launching any process/code."""
    for scene in bpy.data.scenes:
        raw = scene.get("local_repair_preview_record")
        if not raw:
            continue
        try:
            record = json.loads(raw)
            controller = scene.objects.get(adapter.runtime.CONTROLLER)
            original = bpy.data.actions.get(record["original_action"])
            preview_action = bpy.data.actions.get(record["preview_action"])
            if (
                not controller
                or not controller.animation_data
                or controller.animation_data.action != preview_action
            ):
                raise ValueError(
                    "Saved preview controls changed; restore the original Action manually."
                )
            if (
                _json_digest(_action_record(original)) != record["original_digest"]
                or _json_digest(_action_record(preview_action))
                != record["preview_digest"]
            ):
                raise ValueError(
                    "Saved preview was edited; both Actions are retained for manual review."
                )
            controller.animation_data.action = original
            _clear_preview_record(scene)
            collision.mark_collision_stale(scene)
            inspection.mark_dashboard_stale(scene)
            _set_status(
                scene,
                "IDLE",
                "Restored the original Action from a saved transient preview. No worker was started.",
            )
        except (KeyError, ValueError, TypeError) as error:
            _set_status(scene, "STALE", str(error))


_CLASSES = (
    GRAILRepairJoint,
    GRAILRepairSettings,
    GRAILRepair_OT_load,
    GRAILRepair_OT_update_scene,
    GRAILRepair_OT_range,
    GRAILRepair_OT_run,
    GRAILRepair_OT_preview,
    GRAILRepair_OT_restore,
    GRAILRepair_OT_apply,
    GRAILRepair_OT_cancel,
    GRAILREPAIR_PT_panel,
)


def register():
    global _registered
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.grail_repair = PointerProperty(type=GRAILRepairSettings)
    if _load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_load_pre)
    if _load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post)
    _registered = True


def unregister():
    global _registered
    _load_pre(None)
    _registered = False
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_load_pre)
    if _load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post)
    if hasattr(bpy.types.Scene, "grail_repair"):
        del bpy.types.Scene.grail_repair
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
