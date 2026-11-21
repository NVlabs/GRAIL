"""Author an object-only Stage 1 placement correction in an isolated Blender scene.

Run with Blender 4.3: blender -b --enable-autoexec --python this_file.py -- --root DIR
No native human parameter is edited. The human mesh is a frozen display cache;
native translation parameters are recovered from the edited object/control scene.
Source frame zero is Blender frame one. All scene coordinates remain world meters.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def mesh_object(name, vertices, faces, color):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.color = color
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def linear_keys(owner):
    if owner.animation_data and owner.animation_data.action:
        for curve in owner.animation_data.action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"


def make_human(vertices, faces):
    obj = mesh_object("Human_FROZEN_native_cache", vertices[0], faces, (0.10, 0.46, 0.95, 1))
    obj["contract"] = (
        "Unmodified baked native human geometry. Do not edit shape keys or transforms."
    )
    obj.lock_location = obj.lock_rotation = obj.lock_scale = (True, True, True)
    obj.shape_key_add(name="Basis")
    # Relative shape keys with piecewise linear one-hot weights preserve every
    # supplied source-frame vertex exactly to Blender's float32 precision.
    for source_frame in range(1, len(vertices)):
        key = obj.shape_key_add(name=f"FrozenSource_{source_frame:03d}")
        key.data.foreach_set("co", np.asarray(vertices[source_frame], np.float32).ravel())
        for frame, value in ((source_frame, 0.0), (source_frame + 1, 1.0), (source_frame + 2, 0.0)):
            key.value = value
            key.keyframe_insert("value", frame=frame)
        key.value = 0.0
    linear_keys(obj.data.shape_keys)
    return obj


def make_object(vertices, faces, rotations, translations):
    obj = mesh_object("Object_original_mesh_native_track", vertices, faces, (1.0, 0.43, 0.08, 1))
    obj.rotation_mode = "QUATERNION"
    previous = None
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        quaternion = Matrix(rotation.tolist()).to_quaternion()
        if previous is not None and previous.dot(quaternion) < 0:
            quaternion.negate()
        previous = quaternion.copy()
        obj.rotation_quaternion = quaternion
        obj.location = translation.tolist()
        obj.keyframe_insert("location", frame=i + 1)
        obj.keyframe_insert("rotation_quaternion", frame=i + 1)
    linear_keys(obj)
    obj["contract"] = (
        "Original rotation and base translation track; edit only parent placement correction."
    )
    obj.lock_rotation = obj.lock_scale = (True, True, True)
    return obj


def make_control(obj, plan):
    ctrl = bpy.data.objects.new("Stage1_OBJECT_TRANSLATION_EDIT", None)
    bpy.context.collection.objects.link(ctrl)
    ctrl.empty_display_type = "ARROWS"
    ctrl.empty_display_size = 0.3
    ctrl["contract"] = (
        "Editable world translation offset. Human and later object trajectory are protected."
    )
    ctrl["contact_source_frame"] = int(plan["contact_frame"])
    ctrl["recovery_start_source_frame"] = int(plan["recovery_start_frame"])
    ctrl["protected_tail_source_frame"] = int(plan["recovery_end_frame"])
    ctrl["progress"] = 0.0
    ctrl.keyframe_insert('["progress"]', frame=plan["recovery_start_frame"] + 1)
    ctrl["progress"] = 1.0
    ctrl.keyframe_insert('["progress"]', frame=plan["recovery_end_frame"] + 1)
    linear_keys(ctrl)
    ctrl.lock_rotation = ctrl.lock_scale = (True, True, True)
    for axis, label in enumerate("xyz"):
        ctrl["delta_" + label] = 0.0
        driver = ctrl.driver_add("location", axis).driver
        driver.type = "SCRIPTED"
        driver.expression = "d*(1-(10*u**3-15*u**4+6*u**5))"
        for name, prop in (("d", "delta_" + label), ("u", "progress")):
            variable = driver.variables.new()
            variable.name = name
            variable.type = "SINGLE_PROP"
            variable.targets[0].id = ctrl
            variable.targets[0].data_path = '["' + prop + '"]'
    obj.parent = ctrl
    obj.matrix_parent_inverse = Matrix.Identity(4)
    return ctrl


def set_time(source_time):
    frame = float(source_time) + 1.0
    integer = math.floor(frame)
    bpy.context.scene.frame_set(integer, subframe=frame - integer)
    bpy.context.view_layer.update()


def collect_transforms(obj, times):
    rotations, translations, corrections = [], [], []
    ctrl = bpy.data.objects["Stage1_OBJECT_TRANSLATION_EDIT"]
    for time in times:
        set_time(time)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        rotations.append(matrix[:3, :3])
        translations.append(matrix[:3, 3])
        corrections.append(
            np.asarray(ctrl.evaluated_get(bpy.context.evaluated_depsgraph_get()).location)
        )
    return dict(
        source_frames=np.asarray(times, dtype=np.float64),
        obj_R=np.asarray(rotations),
        obj_t=np.asarray(translations),
        delta_world=np.asarray(corrections),
    )


def human_parity(obj, vertices):
    maximum = 0.0
    for i, target in enumerate(vertices):
        set_time(i)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        actual = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get("co", actual.ravel())
        maximum = max(maximum, float(np.max(np.abs(actual - target))))
        evaluated.to_mesh_clear()
    return maximum


def make_camera(name, eye, basis, focal, width=1280):
    data = bpy.data.cameras.new(name)
    data.type = "PERSP"
    data.lens = float(focal) * 36.0 / width
    data.sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.clip_start, data.clip_end = 0.01, 1000.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    matrix = np.eye(4)
    matrix[:3, :3] = basis
    matrix[:3, 3] = eye
    obj.matrix_world = Matrix(matrix.tolist())
    return obj


def configure_views(data, plan):
    rotation = data["camera_R"].reshape(3, 3)
    translation = data["camera_T"].reshape(3)
    eye = -translation @ rotation.T
    basis = rotation @ np.diag([-1.0, 1.0, -1.0])
    height, width = map(int, data["image_size"])
    source = make_camera("Source_calibrated_camera", eye, basis, float(data["focal"]), width)
    frame = int(plan["contact_frame"])
    object_world = data["object_local_scaled"] @ data["obj_R"][frame].T + data["obj_t"][frame]
    combined = np.concatenate([data["human_vertices"][frame], object_world])
    target = 0.5 * (combined.min(0) + combined.max(0))
    angle = np.radians(55.0)
    yaw = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    direction = yaw @ (eye - target)
    novel_eye = target + direction / np.linalg.norm(direction) * 3.8
    quaternion = Vector((target - novel_eye).tolist()).to_track_quat("-Z", "Y")
    oblique = make_camera(
        "Oblique_55deg_camera", novel_eye, np.asarray(quaternion.to_matrix()), 0.82 * width, width
    )
    bpy.context.scene.camera = source
    return source, oblique


def render_stills(root, plan):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0.055, 0.055, 0.055)
    scene.view_settings.view_transform = "Standard"
    ctrl = bpy.data.objects["Stage1_OBJECT_TRANSLATION_EDIT"]
    files = []
    for view, camera_name in (
        ("source", "Source_calibrated_camera"),
        ("oblique", "Oblique_55deg_camera"),
    ):
        scene.camera = bpy.data.objects[camera_name]
        for mode, delta in (("before", [0.0, 0.0, 0.0]), ("after", plan["delta_world"])):
            for label, value in zip("xyz", delta):
                ctrl["delta_" + label] = float(value)
            ctrl.update_tag()
            set_time(plan["contact_frame"])
            path = root / f"blender_contact_{plan['contact_frame']:03d}_{view}_{mode}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            files.append(str(path))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--render-stills", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    root = args.root.resolve()
    input_path, plan_path = root / "edit_inputs.npz", root / "edit_plan.json"
    data, plan = np.load(input_path), json.loads(plan_path.read_text())
    count = len(data["human_vertices"])
    assert (
        0
        <= plan["contact_frame"]
        <= plan["recovery_start_frame"]
        < plan["recovery_end_frame"]
        < count
    )
    assert np.isfinite(np.asarray(plan["delta_world"])).all()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("ReviewWorld")
    scene.frame_start, scene.frame_end = 1, count
    fps = float(plan["fps"])
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    height, width = map(int, data["image_size"])
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene["source_frame_mapping"] = "source_frame = blender_frame - 1"
    scene["input_sha256"] = sha(input_path)
    scene["edit_plan_sha256"] = sha(plan_path)
    scene["human_geometry_contract"] = (
        "Frozen original native cache, not an edited body mesh/model."
    )
    human = make_human(data["human_vertices"], data["human_faces"])
    obj = make_object(
        data["object_local_scaled"], data["object_faces"], data["obj_R"], data["obj_t"]
    )
    ctrl = make_control(obj, plan)
    configure_views(data, plan)
    integers = np.arange(count, dtype=np.float64)
    subframes = np.arange(
        max(0, plan["recovery_start_frame"] - 2),
        min(count - 1, plan["recovery_end_frame"] + 2) + 0.01,
        0.25,
    )
    before = collect_transforms(obj, integers)
    np.savez_compressed(root / "blender_before_integer.npz", **before)
    # This is the actual Blender-native edit: the editable offset control is
    # changed after constructing and exporting the unchanged baseline scene.
    for label, value in zip("xyz", plan["delta_world"]):
        ctrl["delta_" + label] = float(value)
    ctrl.update_tag()
    after = collect_transforms(obj, integers)
    subs = collect_transforms(obj, subframes)
    np.savez_compressed(root / "blender_after_integer.npz", **after)
    np.savez_compressed(root / "blender_after_subframes.npz", **subs)
    hp = human_parity(human, data["human_vertices"])
    set_time(plan["contact_frame"])
    scene.camera = bpy.data.objects["Oblique_55deg_camera"]
    bpy.context.view_layer.objects.active = ctrl
    ctrl.select_set(True)
    blend_path = root / "stage1_object_placement.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene
    reloaded = collect_transforms(bpy.data.objects["Object_original_mesh_native_track"], integers)
    reloaded_sub = collect_transforms(
        bpy.data.objects["Object_original_mesh_native_track"], subframes
    )
    reopened_human_error = human_parity(
        bpy.data.objects["Human_FROZEN_native_cache"], data["human_vertices"]
    )
    np.savez_compressed(root / "blender_reopened_integer.npz", **reloaded)
    np.savez_compressed(root / "blender_reopened_subframes.npz", **reloaded_sub)
    parity = {key: float(np.max(np.abs(after[key] - reloaded[key]))) for key in after}
    sub_parity = {key: float(np.max(np.abs(subs[key] - reloaded_sub[key]))) for key in subs}
    frames = np.arange(count)
    u = np.clip(
        (frames - plan["recovery_start_frame"])
        / (plan["recovery_end_frame"] - plan["recovery_start_frame"]),
        0,
        1,
    )
    weights = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
    expected = data["obj_t"] + weights[:, None] * np.asarray(plan["delta_world"])[None]
    receipt = dict(
        status="BLENDER_EDIT_EXPORTED_UNREVIEWED",
        blender=bpy.app.version_string,
        input_sha256=sha(input_path),
        plan_sha256=sha(plan_path),
        script_sha256=sha(__file__),
        blend=str(blend_path),
        blend_sha256=sha(blend_path),
        frame_count=count,
        source_frame_zero_is_blender_frame=1,
        fps=scene.render.fps / scene.render.fps_base,
        baseline_translation_error_max_m=float(np.max(np.abs(before["obj_t"] - data["obj_t"]))),
        baseline_rotation_error_max=float(np.max(np.abs(before["obj_R"] - data["obj_R"]))),
        human_cached_vertices_max_absolute_error_m=hp,
        reopened_human_cached_vertices_max_absolute_error_m=reopened_human_error,
        analytic_translation_error_max_m=float(np.max(np.abs(after["obj_t"] - expected))),
        object_rotation_edit_max=float(np.max(np.abs(after["obj_R"] - before["obj_R"]))),
        saved_reopened_integer_parity=parity,
        saved_reopened_subframe_parity=sub_parity,
        correction_control="Stage1_OBJECT_TRANSLATION_EDIT delta_x/y/z; quintic driver from linear progress",
        human_subframes="Piecewise linear geometry display only; no claim of native model subframe FK",
        native_export_note="Consumer must preserve original R and copy protected-tail t verbatim; Blender transforms are float32.",
    )
    assert max(parity.values()) == 0 and max(sub_parity.values()) == 0
    assert max(hp, reopened_human_error) < 2e-5
    assert (
        max(
            receipt[name]
            for name in [
                "analytic_translation_error_max_m",
                "baseline_translation_error_max_m",
                "baseline_rotation_error_max",
                "object_rotation_edit_max",
            ]
        )
        < 2e-5
    )
    write_json(root / "blender_export_receipt.json", receipt)
    if args.render_stills:
        receipt["blender_stills"] = render_stills(root, plan)
        write_json(root / "blender_export_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
