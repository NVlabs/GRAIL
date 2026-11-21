"""Blender-only NumPy snapshot/authoring worker; native models run externally.

No embedded script is executed on opening a saved scene. The matching addon
explicitly enables stale-aware snapshot playback. Controls are *local native
rotation corrections*, never world-space IK handles or display-mesh warps.
"""

import base64
import hashlib
import io
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

PREFIX = "GRAILStage2_"
CONTRACT = PREFIX + "Contract.json"
STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def checked_path(value):
    path = Path(value["path"]).resolve(strict=True)
    if digest(path) != value["sha256"]:
        raise ValueError(f"Input hash mismatch: {path}")
    return path


def put_text(name, value):
    block = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    block.clear()
    block.write(value)


def pack(name, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    put_binary(PREFIX + name + ".npz.b64", stream.getvalue())


def put_binary(name, payload):
    # Blender Text.write grows each line incrementally. A geometry cache encoded
    # as one multi-megabyte line is quadratic to insert; MIME's 76-character
    # lines keep insertion bounded. Decoding ignores these newlines and restores
    # the exact original NPZ bytes, including the candidate's hash binding.
    put_text(name, base64.encodebytes(payload).decode("ascii"))


def unpack(name):
    raw = base64.b64decode(bpy.data.texts[PREFIX + name + ".npz.b64"].as_string())
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_arrays(value):
    with np.load(checked_path(value), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if any(a.dtype.kind in "fc" and not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite geometry or scene arrays")
    return arrays


def contract():
    return json.loads(bpy.data.texts[CONTRACT].as_string())


def _quat(rotvec):
    vector = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(vector))
    return Quaternion(vector / angle, angle) if angle > 1e-12 else Quaternion()


def _rotvec(value):
    vector = np.asarray(value, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError("Native correction quaternion must be finite and nonzero")
    vector /= norm
    if vector[0] < 0:
        vector = -vector
    axis_norm = np.linalg.norm(vector[1:])
    if axis_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    return (vector[1:] / axis_norm * (2 * np.arctan2(axis_norm, vector[0]))).tolist()


def authored_plan():
    """Extract actual editable quaternions while preserving declared key topology."""
    data = contract()
    plan = deepcopy(data["plan"])
    for item in data["controls"]:
        ob = bpy.context.scene.objects.get(item["name"])
        if ob is None or ob.rotation_mode != "QUATERNION" or ob.animation_data:
            raise ValueError("Missing, animated or non-quaternion native key control")
        if ob.parent or not np.allclose(ob.location, item["location"], atol=1e-6):
            raise ValueError("Native controls have fixed marker locations; rotate only")
        if not np.allclose(ob.scale, [1, 1, 1], atol=1e-6):
            raise ValueError("Scaling a native key control is unsupported")
        delta = _rotvec(ob.rotation_quaternion)
        window = plan["windows"][item["window"]]
        key = window["keys"][item["key"]]
        if item["protected"] and not np.allclose(delta, key["delta_rotvec"], atol=1e-6):
            raise ValueError("Protected endpoint/contact key changed")
        angle = float(np.linalg.norm(delta))
        limit = float(window["max_angle_rad"])
        if angle > limit + 1e-6:
            raise ValueError("Native correction exceeds its declared angular bound")
        if angle > limit:
            # Blender stores float32 quaternions. Snap representational excess
            # at the boundary without loosening the native angular contract.
            delta = (np.asarray(delta) * (limit / angle)).tolist()
        key["delta_rotvec"] = delta
    return plan


def validate_scene_binding(data):
    """Reject world-space mesh warps and camera edits posing as native edits."""
    scene = bpy.context.scene
    for name in ("BeforeHuman", "AfterHuman"):
        ob = scene.objects.get(PREFIX + name)
        if (
            ob is None
            or ob.type != "MESH"
            or not np.allclose(ob.matrix_world, np.eye(4), atol=1e-6)
        ):
            raise ValueError("Original native human mesh transforms must remain unchanged")
        if ob.modifiers or ob.constraints or ob.animation_data:
            raise ValueError("Native human mesh modifiers/constraints/animation are unsupported")
    source = scene.objects.get(PREFIX + "SourceCamera")
    camera = data["source_camera"]
    if source is None or not np.allclose(source.matrix_world, camera["matrix_world"], atol=1e-6):
        raise ValueError("Calibrated source camera transform changed")
    if source.data.type != "PERSP" or source.data.sensor_fit != "HORIZONTAL":
        raise ValueError("Calibrated source camera projection changed")
    for key in ("lens", "sensor_width", "shift_x", "shift_y"):
        if not np.isclose(getattr(source.data, key), camera[key], atol=1e-6):
            raise ValueError("Calibrated source camera intrinsics changed")
    for key, value in camera["render"].items():
        if not np.isclose(getattr(scene.render, key), value, atol=1e-6):
            raise ValueError("Calibrated source camera resolution/aspect changed")


def validate_original_meshes(before, after, arrays):
    for name, geometry in (
        ("BeforeHuman", before),
        ("AfterHuman", after),
        ("Object", {"vertices": arrays["object_vertices"][None], "faces": arrays["object_faces"]}),
    ):
        ob = bpy.context.scene.objects.get(PREFIX + name)
        if ob is None or ob.type != "MESH" or ob.mode != "OBJECT":
            raise ValueError("Native snapshots require original meshes in Object Mode")
        if ob.modifiers or ob.constraints or ob.animation_data:
            raise ValueError("Original native meshes cannot have modifiers or animation")
        faces = geometry["faces"]
        if len(ob.data.vertices) != geometry["vertices"].shape[1] or len(ob.data.polygons) != len(
            faces
        ):
            raise ValueError("Original native mesh topology changed")
        counts = np.empty(len(faces), dtype=np.int32)
        ob.data.polygons.foreach_get("loop_total", counts)
        if np.any(counts != 3):
            raise ValueError("Original native triangular mesh topology changed")
        indices = np.empty(faces.size, dtype=np.int32)
        ob.data.loops.foreach_get("vertex_index", indices)
        if not np.array_equal(indices.reshape(-1, 3), faces):
            raise ValueError("Original native mesh face correspondence changed")
        if name == "Object":
            positions = np.empty(geometry["vertices"].size, dtype=np.float32)
            ob.data.vertices.foreach_get("co", positions)
            expected = arrays["object_vertices"] * arrays.get("object_scale", 1.0)
            if not np.allclose(positions.reshape(-1, 3), expected, atol=1e-6):
                raise ValueError("Original native object mesh changed")


def _plan_equal(left, right):
    a, b = deepcopy(left), deepcopy(right)
    for wa, wb in zip(a["windows"], b["windows"], strict=True):
        for ka, kb in zip(wa["keys"], wb["keys"], strict=True):
            if not np.allclose(ka.pop("delta_rotvec"), kb.pop("delta_rotvec"), atol=1e-6):
                return False
    return a == b


def disable_snapshots():
    bound_scene = STATE.get("scene")
    if bound_scene is not None:
        _hide(bound_scene, "DISABLED: enable verified native snapshots")
    for handlers in (bpy.app.handlers.frame_change_post, bpy.app.handlers.depsgraph_update_post):
        if snapshot_update in handlers:
            handlers.remove(snapshot_update)
    STATE.clear()


def _hide(scene, reason):
    for name in ("BeforeHuman", "AfterHuman", "Object"):
        ob = scene.objects.get(PREFIX + name)
        if ob:
            ob.hide_viewport = True
            ob.hide_render = True
    scene["grail_stage2_snapshot_status"] = reason


def enable_snapshots():
    disable_snapshots()
    data = contract()
    validate_scene_binding(data)
    STATE.update(
        scene=bpy.context.scene,
        contract=data,
        before=unpack("before"),
        after=unpack("after"),
        scene_arrays=unpack("scene"),
        updating=False,
    )
    for handlers in (bpy.app.handlers.frame_change_post, bpy.app.handlers.depsgraph_update_post):
        handlers.append(snapshot_update)
    snapshot_update(bpy.context.scene)


def snapshot_update(scene, *unused):
    """Only reveal exact cached observations with unchanged bound native controls."""
    if STATE.get("scene") != scene or STATE.get("updating"):
        return
    STATE["updating"] = True
    try:
        validate_scene_binding(STATE["contract"])
        validate_original_meshes(STATE["before"], STATE["after"], STATE["scene_arrays"])
        source_frame = float(scene.frame_current) + float(scene.frame_subframe) - 1
        if not _plan_equal(authored_plan(), STATE["contract"]["plan"]):
            _hide(scene, "STALE: keys changed; save and run external native Refresh")
            return
        frames = STATE["after"]["frames"]
        matches = np.flatnonzero(np.isclose(frames, source_frame, atol=1e-6, rtol=0))
        if len(matches) != 1:
            _hide(scene, "STALE: uncached source frame; external native Refresh required")
            return
        show_snapshot("after", int(matches[0]), STATE["after"], STATE["scene_arrays"])
        scene["grail_stage2_snapshot_status"] = "UNREVIEWED native snapshot; exact source frame"
    except Exception as error:
        _hide(scene, "STALE: " + str(error))
    finally:
        STATE["updating"] = False


def mesh(name, vertices, faces, color):
    data = bpy.data.meshes.new(PREFIX + name)
    data.from_pydata(vertices.tolist(), [], faces.tolist())
    data.update()
    ob = bpy.data.objects.new(PREFIX + name, data)
    bpy.context.scene.collection.objects.link(ob)
    material = bpy.data.materials.new(PREFIX + name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
    material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
    data.materials.append(material)
    ob.color = (*color, 1)
    return ob


def show_snapshot(phase, index, geometry, scene_arrays):
    scene = bpy.context.scene
    name = "BeforeHuman" if phase == "before" else "AfterHuman"
    for other in ("BeforeHuman", "AfterHuman"):
        ob = scene.objects[PREFIX + other]
        hidden = other != name
        ob.hide_render = hidden
        ob.hide_viewport = hidden
    ob = scene.objects[PREFIX + name]
    ob.data.vertices.foreach_set("co", np.asarray(geometry["vertices"][index]).reshape(-1))
    ob.data.update()
    source_frame = float(geometry["frames"][index])
    if not source_frame.is_integer():
        # Fractional native observations require evaluated object transforms.
        positions = scene_arrays.get("observation_obj_t")
        rotations = scene_arrays.get("observation_obj_R")
        if positions is None or rotations is None:
            raise ValueError("Fractional observations require observation_obj_R/obj_t")
        r, t = rotations[index], positions[index]
    else:
        r, t = scene_arrays["obj_R"][int(source_frame)], scene_arrays["obj_t"][int(source_frame)]
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = r, t
    object_ob = scene.objects[PREFIX + "Object"]
    object_ob.matrix_world = Matrix(matrix.tolist())
    object_ob.hide_render = object_ob.hide_viewport = False


def _camera(name, matrix):
    data = bpy.data.cameras.new(PREFIX + name)
    ob = bpy.data.objects.new(PREFIX + name, data)
    bpy.context.scene.collection.objects.link(ob)
    ob.matrix_world = Matrix(matrix.tolist())
    return ob


def cameras(arrays, geometry):
    cv_to_blender = np.diag([1, -1, -1, 1])
    source = _camera("SourceCamera", arrays["camera_matrix_world"] @ cv_to_blender)
    width, height = int(arrays["image_width"]), int(arrays["image_height"])
    k = arrays["camera_intrinsics"]
    if abs(float(k[0, 1])) > 1e-8 or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Camera intrinsics require positive focal length and zero skew")
    source.data.sensor_fit = "HORIZONTAL"
    source.data.sensor_width = 36
    source.data.lens = float(k[0, 0]) * 36 / width
    source.data.shift_x = (width / 2 - float(k[0, 2])) / width
    pixel_ratio = float(k[0, 0] / k[1, 1])
    source.data.shift_y = (float(k[1, 2]) - height / 2) * pixel_ratio / width
    source.data.clip_start, source.data.clip_end = 0.01, 1000
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1, pixel_ratio
    center = geometry["vertices"].reshape(-1, 3).mean(axis=0)
    extent = max(float(np.ptp(geometry["vertices"].reshape(-1, 3), axis=0).max()), 0.5)
    oblique = _camera("ObliqueCamera", np.eye(4))
    # Native world axes are arbitrary: offset relative to the calibrated camera.
    offset = arrays["camera_matrix_world"][:3, :3] @ np.array([1.3, -0.45, -2.0])
    position = center + extent * offset
    source_up = -arrays["camera_matrix_world"][:3, 1]
    view_z = position - center
    view_z /= np.linalg.norm(view_z)
    right = np.cross(source_up, view_z)
    right /= np.linalg.norm(right)
    view_up = np.cross(view_z, right)
    oblique_matrix = np.eye(4)
    oblique_matrix[:3, :3] = np.column_stack((right, view_up, view_z))
    oblique_matrix[:3, 3] = position
    oblique.matrix_world = Matrix(oblique_matrix.tolist())
    oblique.data.type = "ORTHO"
    oblique.data.ortho_scale = extent * 1.8 * max(1, height / width)
    return source, oblique


def contact_overlays(config):
    """Explicit world-space material points/normals; diagnostic overlays only."""
    result = []
    for index, sample in enumerate(config.get("samples", [])):
        phase = sample.get("phase", "after")
        if phase not in ("before", "after"):
            raise ValueError("Contact overlay phase must be before or after")
        for role, color in (("human", (0.05, 0.7, 0.9)), ("object", (0.9, 0.15, 0.4))):
            point = np.asarray(sample[role + "_point"], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Contact overlay requires finite world-space points")
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=8, ring_count=4, radius=0.008, location=point.tolist()
            )
            marker = bpy.context.object
            marker.name = PREFIX + f"Contact_{index:04d}_{role}"
            marker["diagnostic"] = "Authored material contact hypothesis; not native geometry"
            material = bpy.data.materials.new(marker.name)
            material.diffuse_color = (*color, 1)
            material.use_nodes = True
            inputs = material.node_tree.nodes["Principled BSDF"].inputs
            inputs["Base Color"].default_value = (*color, 1)
            marker.data.materials.append(material)
            objects = [marker]
            if role + "_normal" in sample:
                normal = np.asarray(sample[role + "_normal"], dtype=float)
                if (
                    normal.shape != (3,)
                    or not np.isfinite(normal).all()
                    or np.linalg.norm(normal) < 1e-10
                ):
                    raise ValueError("Contact overlay normals must be finite and nonzero")
                tip = point + 0.05 * normal / np.linalg.norm(normal)
                curve = bpy.data.curves.new(marker.name + "_normal", "CURVE")
                curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", 0.002, 1
                spline = curve.splines.new("POLY")
                spline.points.add(1)
                spline.points[0].co, spline.points[1].co = (*point, 1), (*tip, 1)
                normal_ob = bpy.data.objects.new(curve.name, curve)
                bpy.context.scene.collection.objects.link(normal_ob)
                curve.materials.append(material)
                objects.append(normal_ob)
            for ob in objects:
                ob.hide_viewport = ob.hide_render = True
                result.append((ob, phase, float(sample["source_frame"])))
    return result


def validate_geometry(before, after, scene_arrays, request):
    observations = np.asarray(request["observation_frames"], dtype=float)
    if (
        observations.ndim != 1
        or not len(observations)
        or len(set(observations.tolist())) != len(observations)
    ):
        raise ValueError("Observation frames must be a nonempty unique list")
    frames = np.asarray(after["frames"], dtype=float)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("Cached geometry frames must be unique and sorted")
    if not set(observations.tolist()).issubset(frames.tolist()):
        raise ValueError("Native geometry must contain every rendered observation")
    for geometry in (before, after):
        v, j, f = (geometry[name] for name in ("vertices", "joints", "faces"))
        if v.ndim != 3 or v.shape[0] != len(frames) or v.shape[-1] != 3:
            raise ValueError("Expected native vertices [observations,vertices,3]")
        if j.ndim != 3 or j.shape[0] != len(frames) or j.shape[-1] != 3:
            raise ValueError("Expected native joints [observations,joints,3]")
        if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
            raise ValueError("Expected original triangular mesh faces")
        if f.min() < 0 or f.max() >= v.shape[1] or not np.array_equal(geometry["frames"], frames):
            raise ValueError("Native topology/frame binding mismatch")
    if not np.array_equal(before["faces"], after["faces"]):
        raise ValueError("Original native topology changed")
    if request.get("camera_convention") != "opencv_camera_to_world":
        raise ValueError("Declare camera_convention=opencv_camera_to_world")
    if scene_arrays["camera_matrix_world"].shape != (4, 4):
        raise ValueError("Camera matrix must be [4,4]")
    if frames.min() < 0 or frames.max() > len(scene_arrays["obj_t"]) - 1:
        raise ValueError("Observation frame outside original source timeline")


def create_controls(plan, geometry):
    controls = []
    frame_map = {float(frame): index for index, frame in enumerate(geometry["frames"])}
    for wi, window in enumerate(plan["windows"]):
        joint = int(window["joint"])
        for ki, key in enumerate(window["keys"]):
            name = PREFIX + f"Key_w{wi:03d}_j{joint:03d}_k{ki:03d}"
            ob = bpy.data.objects.new(name, None)
            bpy.context.scene.collection.objects.link(ob)
            ob.empty_display_type = "ARROWS"
            ob.empty_display_size = 0.065
            ob.show_in_front = True
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = _quat(key["delta_rotvec"])
            if float(key["frame"]) not in frame_map:
                raise ValueError("Native geometry is missing a declared authoring key frame")
            fi = frame_map[float(key["frame"])]
            if not 0 <= joint < geometry["joints"].shape[1]:
                raise ValueError("Native key joint outside original joint array")
            ob.location = geometry["joints"][fi, joint].tolist()
            protected = key["frame"] in (window["start"], window["end"]) or key[
                "frame"
            ] in window.get("protected_frames", [])
            ob.lock_location = ob.lock_scale = (True, True, True)
            ob.lock_rotation = (protected, protected, protected)
            ob.lock_rotation_w = protected
            ob["source_frame"], ob["blender_frame"] = key["frame"], key["frame"] + 1
            ob["native_joint"], ob["protected"] = joint, protected
            ob["meaning"] = "Local native delta quaternion; fixed marker, not IK"
            controls.append(
                dict(name=name, window=wi, key=ki, protected=protected, location=list(ob.location))
            )
    return controls


def render_pairs(out, before, after, arrays, camera_objects, config, observation_frames, overlays):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(config.get("samples", 8))
    if not 1 <= scene.cycles.samples <= 64:
        raise ValueError("CPU render samples must be in [1,64]")
    scene.render.resolution_percentage = int(config.get("resolution_percentage", 100))
    if not 1 <= scene.render.resolution_percentage <= 100:
        raise ValueError("resolution_percentage must be in [1,100]")
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_compositing = scene.render.use_sequencer = False
    scene.world = bpy.data.worlds.new(PREFIX + "World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    records = []
    for phase, geometry in (("before", before), ("after", after)):
        for index, frame in enumerate(geometry["frames"]):
            if float(frame) not in observation_frames:
                continue
            blender_frame = float(frame) + 1
            scene.frame_set(int(blender_frame), subframe=blender_frame % 1)
            show_snapshot(phase, index, geometry, arrays)
            for ob, overlay_phase, overlay_frame in overlays:
                ob.hide_render = not (phase == overlay_phase and float(frame) == overlay_frame)
            for cam in camera_objects:
                path = out / f"{phase}_{index:04d}_{cam.name}.png"
                scene.camera, scene.render.filepath = cam, str(path)
                bpy.ops.render.render(write_still=True)
                if not path.is_file():
                    raise RuntimeError("Blender failed to write native observation")
                records.append(
                    dict(
                        phase=phase,
                        stage=phase,
                        source_frame=float(frame),
                        camera=cam.name,
                        contact_overlays=any(not ob.hide_render for ob, _, _ in overlays),
                        **binding(path),
                    )
                )
    for ob, _, _ in overlays:
        ob.hide_render = True
    return records


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect(out):
    data = contract()
    validate_scene_binding(data)
    validate_original_meshes(unpack("before"), unpack("after"), unpack("scene"))
    plan = authored_plan()
    path = out / "authored_plan.json"
    _write_json(path, plan)
    native_path = out / "previous_candidate.npz"
    native_text = bpy.data.texts[PREFIX + "Candidate.npz.b64"].as_string()
    native_path.write_bytes(base64.b64decode(native_text))
    if digest(native_path) != data["candidate_motion"]["sha256"]:
        raise ValueError("Saved candidate native bytes failed binding")
    return dict(
        status="ok",
        review_status="UNREVIEWED",
        plan=plan,
        parent_motion=binding(native_path),
        bindings=data.get("bindings", {}),
        artifacts=dict(plan=binding(path), parent_motion=binding(native_path)),
        diagnostics=dict(
            controls_changed=not _plan_equal(plan, data["plan"]),
            cached_geometry_is_authoritative=False,
            source_to_blender_offset=1,
        ),
    )


def run(request):
    if request.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    out = Path(request["output_dir"]).resolve()
    if request.get("input_blend"):
        bpy.ops.wm.open_mainfile(filepath=str(checked_path(request["input_blend"])))
        if request.get("mode") == "inspect":
            return inspect(out)
        old = contract()
        if request.get("bindings", {}) != old.get("bindings", {}):
            raise ValueError("Continuation scene/model/source bindings changed")
        # Work in a new scene inside the isolated copy; preserve unrelated user data.
        new_scene = bpy.data.scenes.new(PREFIX + "Refreshed")
        bpy.context.window.scene = new_scene
        for ob in list(bpy.data.objects):
            if ob.name.startswith(PREFIX):
                bpy.data.objects.remove(ob, do_unlink=True)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    before, after, arrays = (
        read_arrays(request[name]) for name in ("before_geometry", "after_geometry", "scene")
    )
    validate_geometry(before, after, arrays, request)
    checked_path(request["parent_motion"])
    native = checked_path(request["candidate_motion"])
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(arrays["obj_t"])
    fps = float(request.get("fps", 30))
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    for name, geometry, color in (
        ("BeforeHuman", before, (0.28, 0.48, 0.7)),
        ("AfterHuman", after, (0.35, 0.7, 0.18)),
    ):
        mesh(name, geometry["vertices"][0], geometry["faces"], color)
    scale = arrays.get("object_scale", np.asarray(1.0))
    mesh("Object", arrays["object_vertices"] * scale, arrays["object_faces"], (0.55, 0.34, 0.18))
    camera_objects = cameras(arrays, after)
    controls = create_controls(request["plan"], after)
    data = dict(
        schema_version=1,
        plan=request["plan"],
        controls=controls,
        source_to_blender_offset=1,
        parent_motion=request["parent_motion"],
        candidate_motion=request["candidate_motion"],
        bindings=request.get("bindings", {}),
        geometry_bindings={name: request[name] for name in ("before_geometry", "after_geometry")},
        model_type=request.get("model_type", "undeclared"),
        review_status="UNREVIEWED",
    )
    source = camera_objects[0]
    data["source_camera"] = dict(
        matrix_world=[list(row) for row in source.matrix_world],
        **{
            key: getattr(source.data, key) for key in ("lens", "sensor_width", "shift_x", "shift_y")
        },
    )
    data["source_camera"]["render"] = {
        key: getattr(scene.render, key)
        for key in ("resolution_x", "resolution_y", "pixel_aspect_x", "pixel_aspect_y")
    }
    put_text(CONTRACT, json.dumps(data, allow_nan=False))
    for name, values in (("before", before), ("after", after), ("scene", arrays)):
        pack(name, values)
    put_binary(PREFIX + "Candidate.npz.b64", native.read_bytes())
    put_text(
        PREFIX + "README.txt",
        "UNREVIEWED native Stage 2 observations.\n"
        "Enable snapshots in the installed GRAIL Stage 2 panel after opening.\n"
        "Source frame 0 = Blender frame 1. Exact native observations only.\n"
        "Rotate selected static local-delta keys, save a NEW .blend, then Refresh.\n"
        "Keys are independent source-frame controls; no Blender interpolation.\n"
        "Hidden/stale geometry requires external native evaluation. No Torch in Blender.\n",
    )
    overlays = contact_overlays(request.get("contact_overlay", {}))
    renders = []
    if request.get("render", True):
        renders = render_pairs(
            out,
            before,
            after,
            arrays,
            camera_objects,
            request.get("render_config", {}),
            request["observation_frames"],
            overlays,
        )
    scene.camera = camera_objects[0]
    first_frame = float(after["frames"][0]) + 1
    scene.frame_set(int(first_frame), subframe=first_frame % 1)
    show_snapshot("after", 0, after, arrays)
    _hide(scene, "DISABLED after open: enable verified snapshots in GRAIL Stage 2 panel")
    checkpoint = out / "UNREVIEWED_authoring.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(checkpoint), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(checkpoint))
    result = inspect(out)
    if not _plan_equal(result["plan"], request["plan"]):
        raise ValueError("Save/reopen changed authored native controls")
    reopened = unpack("after")
    for key in after:
        if not np.array_equal(reopened[key], after[key]):
            raise ValueError("Save/reopen changed cached native observation arrays")
    actual = np.empty(after["vertices"][0].size, dtype=np.float32)
    bpy.context.scene.objects[PREFIX + "AfterHuman"].data.vertices.foreach_get("co", actual)
    error = float(np.max(np.abs(actual.reshape(-1, 3) - after["vertices"][0])))
    if error > 1e-5:
        raise ValueError("Saved Blender mesh differs from native snapshot")
    result.update(
        checkpoint=str(checkpoint), renders=renders, parent_motion=request["parent_motion"]
    )
    result["artifacts"]["checkpoint"] = binding(checkpoint)
    result["diagnostics"].update(
        saved_reopened=True,
        native_bytes_preserved=True,
        snapshot_mesh_max_error=error,
        snapshot_arrays_exact=True,
        source_camera_convention="opencv_camera_to_world",
        cpu_only=True,
    )
    result["diagnostics"]["contact_overlay_samples"] = len(
        request.get("contact_overlay", {}).get("samples", [])
    )
    return result


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    try:
        response = run(request)
    except Exception as error:
        _write_json(
            Path(request["output_dir"]) / "report.json",
            dict(status="error", error=str(error), traceback=traceback.format_exc()),
        )
        raise
    _write_json(Path(request["output_dir"]) / "report.json", response)


if __name__ == "__main__":
    main()
