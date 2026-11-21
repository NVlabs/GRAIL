"""Blender-only worker for render_sequence; no embedded scene Python executes."""
from pathlib import Path
import json
import runpy
import shutil
import sys
import time
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")))["bootstrap"]()
from agentic_retargeter.blender_worker import Preview, load_runtime
from agentic_retargeter.static_scene import assert_static_scene_equal
from agentic_retargeter.render_sequence import (
    cache_lookup, cache_store, code_fingerprint, json_write, sample_key, sha256,
)


def run(request):
    started = time.perf_counter()
    if not (4, 2, 0) <= bpy.app.version < (4, 4, 0):
        raise ValueError("Use Blender 4.2/4.3; tested with 4.3.2")
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise ValueError("Use --disable-autoexec")
    for key in ("blend", "blender", "native"):
        if key in request["bindings"] and sha256(request[key]) != request["bindings"][key]:
            raise ValueError("Input hash mismatch: " + key)
    if code_fingerprint() != request["bindings"]["code"]:
        raise ValueError("Implementation hash mismatch")
    # Stateful simulation and arbitrary callbacks cannot reuse pose-only images.
    for name in ("frame_change_pre", "frame_change_post", "render_pre", "render_post"):
        if getattr(bpy.app.handlers, name):
            raise ValueError("Scene evaluation handlers are unsupported")
    if bpy.context.scene.rigidbody_world:
        raise ValueError("Bake/remove scene simulation before a native preview")
    for obj in bpy.data.objects:
        if obj.particle_systems or (obj.animation_data and obj.animation_data.drivers):
            raise ValueError("Stateful particles/drivers are unsupported: " + obj.name)
        if any(m.type in {"CLOTH", "FLUID", "SOFT_BODY", "DYNAMIC_PAINT", "NODES"}
               for m in obj.modifiers):
            raise ValueError("Unsupported simulation/procedural modifier: " + obj.name)
    dependencies = {}
    for name in bpy.utils.blend_paths(absolute=True, packed=False):
        path = Path(name).resolve()
        if not path.is_file():
            raise ValueError("Pack missing, sequenced or tiled external assets: " + name)
        dependencies[str(path)] = sha256(path)
    runtime = load_runtime()
    native = runtime.STATE["native"]
    if request.get("native"):
        with np.load(request["native"], allow_pickle=False) as archive:
            candidate = {key: archive[key] for key in archive.files}
        if candidate["qpos"].shape != native["qpos"].shape or not np.isfinite(candidate["qpos"]).all():
            raise ValueError("Candidate qpos must match the template shape and be finite")
        assert_static_scene_equal(native, candidate)
        # A stable template is the cache contract: only native qpos can change.
        for key in ("fps", "obj_R", "obj_t", "object_scale", "scene_scale", "height_scale",
                    "height_offset", "pelvis_rotation"):
            if key in native or key in candidate:
                if key not in candidate or key not in native or not np.array_equal(candidate[key], native[key]):
                    raise ValueError("Candidate changes a template field: " + key)
        qpos = candidate["qpos"]
        runtime.validate_source_quaternions(dict(native, qpos=qpos), runtime.STATE["model"])
    else:
        qpos = np.array([runtime.q_at(f) for f in range(len(native["qpos"]))])
    frames = request["frames"]
    if frames[-1] >= len(qpos):
        raise ValueError("Native frame outside sequence")
    # This is an exported integer-sample preview, not a curve authoring operation.
    render_runtime = SimpleNamespace(STATE=runtime.STATE, unpack=runtime.unpack,
                                     q_at=lambda f: qpos[int(f)].copy())
    spec = request.get("spec")
    if spec is None:
        spec = json.loads(bpy.data.texts["FreshWorkerSpec.json"].as_string())
    preview = Preview(spec, render_runtime)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = request.get("resolution", [480, 480])
    scene.render.resolution_percentage = 100
    scene.render.threads_mode, scene.render.threads = "FIXED", 2
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    cameras = {}
    for view in request["views"]:
        if "camera" in view:
            camera = bpy.data.objects.get(view["camera"])
            if camera is None or camera.type != "CAMERA":
                raise ValueError("Missing camera: " + view["camera"])
        else:
            if view["follow_wrist"] not in preview.bindings:
                raise ValueError("Unknown wrist body")
            camera = bpy.data.objects.new("Render_" + view["name"], bpy.data.cameras.new("RenderLens"))
            scene.collection.objects.link(camera)
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = view.get("ortho_scale", 0.55)
        cameras[view["name"]] = camera

    def set_view(view):
        camera = cameras[view["name"]]
        if "follow_wrist" in view:
            wrist = bpy.data.objects[preview.bindings[view["follow_wrist"]]]
            center = (wrist.matrix_world.translation +
                      bpy.data.objects[preview.object_name].matrix_world.translation) * 0.5
            camera.location = center + Vector(view["direction"]).normalized()
            camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera

    # Background Blender creates its graphics context lazily. A tiny, unexported
    # warmup identifies the actual backend before any image cache is consulted.
    preview.update(frames[0])
    set_view(request["views"][0])
    resolution = (scene.render.resolution_x, scene.render.resolution_y)
    scene.render.resolution_x = scene.render.resolution_y = 16
    tick = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    warmup = time.perf_counter() - tick
    import gpu
    device = dict(renderer=gpu.platform.renderer_get(), vendor=gpu.platform.vendor_get(),
                  version=gpu.platform.version_get())
    scene.render.resolution_x, scene.render.resolution_y = resolution
    if request.get("require_hardware") and any(
            text in device["renderer"].lower() for text in ("llvmpipe", "softpipe", "software")):
        raise ValueError("Hardware rendering required but actual backend is " + device["renderer"])
    context = dict(schema_version=1, blend=request["bindings"]["blend"],
                   blender=request["bindings"]["blender"], code=request["bindings"]["code"],
                   blender_build=bpy.app.build_hash.decode(), device=device,
                   dependencies=dependencies, resolution=resolution, spec=spec,
                   render_policy="WORKBENCH_RGB_NO_COMPOSITOR_V1")
    rows, pending = [], []
    output = Path(request["output"])
    worst = 0.0
    for frame in frames:
        preview.update(frame)
        positions, rotations = preview.fk.forward(qpos[frame])
        for view in request["views"]:
            set_view(view)
            key = sample_key(context, frame, qpos[frame], view)
            name = "%s_%04d.png" % (view["name"], frame)
            destination = output / name
            hit = cache_lookup(request["cache"], key)
            tick = time.perf_counter()
            if hit:
                image, evidence = hit
                shutil.copyfile(image, destination)
            else:
                scene.render.filepath = str(destination)
                bpy.ops.render.render(write_still=True)
            # Keep the original post-render FK check on every new image. Cache
            # receipts carry the same check bound to identical inputs/backend.
            error = 0.0
            for body, body_name in enumerate(preview.fk.names[1:], start=1):
                matrix = np.array(bpy.data.objects[preview.bindings[body_name]].matrix_world)
                error = max(error, float(abs(matrix[:3, 3] - positions[body]).max()),
                            float(abs(matrix[:3, :3] - rotations[body]).max()))
            object_matrix = np.array(bpy.data.objects[preview.object_name].matrix_world)
            scale = np.broadcast_to(np.asarray(native["object_scale"]).reshape(-1), (3,))
            error = max(error, float(abs(object_matrix[:3, 3] - native["obj_t"][frame]).max()),
                        float(abs(object_matrix[:3, :3] - native["obj_R"][frame] @ np.diag(scale)).max()))
            if error >= 2e-6:
                raise ValueError("Post-render native FK mismatch: " + str(error))
            worst = max(worst, error)
            if not hit:
                evidence = dict(sha256=sha256(destination), fk_error=error)
                pending.append((key, destination, evidence))
            rows.append(dict(frame=frame, view=view["name"], file=name, cached=bool(hit), key=key,
                             sha256=evidence["sha256"], seconds=time.perf_counter()-tick))
    for path, digest in dependencies.items():
        if sha256(path) != digest:
            raise ValueError("External scene asset changed while rendering: " + path)
    for key in ("blend", "blender", "native"):
        if key in request["bindings"] and sha256(request[key]) != request["bindings"][key]:
            raise ValueError("Input changed while rendering: " + key)
    if code_fingerprint() != request["bindings"]["code"]:
        raise ValueError("Implementation changed while rendering")
    for key, destination, evidence in pending:
        cache_store(request["cache"], key, destination, evidence)
    result = dict(device=device, warmup_seconds=warmup, worker_seconds=time.perf_counter()-started,
                  native_fps=float(native["fps"]), native_frames=len(qpos), after_render_fk_error=worst,
                  images=rows, static_scene=preview.static_scene, motion_accepted=False)
    json_write(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "images"}))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text()))
