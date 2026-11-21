"""Replay exported native HOI geometry inside an existing Blender scene.

Run with Blender's Python, not the model runtime. This imports a hash-bound
vertex animation for inspection; it does not fit motion or create a body rig.
Source frame zero maps to Blender frame one. No embedded Python is needed for
playback of the saved .blend file.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

PREFIX = "GRAILReplay_"


def binding(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest()}


def linear_keys(owner):
    if owner.animation_data and owner.animation_data.action:
        for curve in owner.animation_data.action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"


def mesh_sequence(name, vertices, faces, color):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices[0].tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.color = color
    material = bpy.data.materials.new(name + "Material")
    material.diffuse_color = color
    obj.data.materials.append(material)
    for face in mesh.polygons:
        face.use_smooth = True
    obj["semantics"] = "Unmodified native mesh replay; not a new rig or fitted result"
    obj.lock_location = obj.lock_rotation = obj.lock_scale = (True, True, True)
    obj.shape_key_add(name="Basis")
    for source_frame in range(1, len(vertices)):
        key = obj.shape_key_add(name=f"Source_{source_frame:04d}")
        key.data.foreach_set("co", np.asarray(vertices[source_frame], np.float32).ravel())
        for frame, value in ((source_frame, 0.0), (source_frame + 1, 1.0),
                             (source_frame + 2, 0.0)):
            key.value = value
            key.keyframe_insert("value", frame=frame)
        key.value = 0.0
    linear_keys(obj.data.shape_keys)
    return obj


def camera(name, eye, basis, focal, width):
    data = bpy.data.cameras.new(name)
    data.type = "PERSP"
    data.lens = float(focal) * 36.0 / width
    data.sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.clip_start, data.clip_end = 0.01, 1000.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = basis, eye
    obj.matrix_world = Matrix(matrix.tolist())
    return obj


def validate_geometry(data):
    frames = len(data["human_vertices"])
    if frames == 0 or len(data["object_vertices"]) != frames:
        raise ValueError("Human/object sequence lengths differ or are empty")
    for kind in ("human", "object"):
        vertices, faces = data[kind + "_vertices"], data[kind + "_faces"]
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid {kind} vertices")
        if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
                or faces.min() < 0 or faces.max() >= vertices.shape[1]):
            raise ValueError(f"Invalid {kind} triangular faces")
    rotation = data["camera_R"].reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Camera rotation is not orthonormal")
    focal = np.asarray(data["focal"]).reshape(-1)
    if len(focal) != 1 or not np.isfinite(focal).all() or focal[0] <= 0:
        raise ValueError("This importer currently requires the native scalar focal length")
    return frames


def verify(data):
    scene = bpy.context.scene
    count = len(data["human_vertices"])
    errors = {}
    for kind in ("human", "object"):
        obj = bpy.data.objects[PREFIX + kind]
        maximum = 0.0
        for index, target in enumerate(data[kind + "_vertices"]):
            scene.frame_set(index + 1)
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            actual = np.empty_like(target)
            mesh.vertices.foreach_get("co", actual.ravel())
            matrix = np.asarray(evaluated.matrix_world)
            actual = actual @ matrix[:3, :3].T + matrix[:3, 3]
            maximum = max(maximum, float(np.abs(actual - target).max()))
            evaluated.to_mesh_clear()
        errors[kind + "_max_vertex_error_m"] = maximum
        if maximum > 1e-5:
            raise ValueError(f"Blender replay changed {kind} geometry: {maximum}")
    # Verify the calibrated camera independently against the native row-vector
    # PyTorch3D projection at fixed vertices across the sequence.
    rotation, translation = data["camera_R"].reshape(3, 3), data["camera_T"].reshape(3)
    height, width = map(int, data["image_size"])
    focal = float(np.asarray(data["focal"]).item())
    source = bpy.data.objects[PREFIX + "source"]
    projection_error = 0.0
    for index in sorted({0, count // 2, count - 1}):
        points = np.concatenate([data["human_vertices"][index, ::47],
                                 data["object_vertices"][index, ::113]])
        view = points @ rotation + translation
        expected = -view[:, :2] / view[:, 2:] * focal + np.array([width, height]) / 2
        actual = []
        for point in points:
            coordinate = world_to_camera_view(scene, source, Vector(point.tolist()))
            actual.append([coordinate.x * width, (1 - coordinate.y) * height])
        projection_error = max(projection_error, float(np.abs(np.asarray(actual) - expected).max()))
    errors["source_projection_max_error_px"] = projection_error
    if projection_error > 0.02:
        raise ValueError(f"Camera projection mismatch: {projection_error} px")
    errors["frames_checked"] = count
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--background-scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--model-label", default="Native human")
    parser.add_argument("--quality-status", default="UNREVIEWED")
    parser.add_argument("--frames", default="0", help="Comma-separated source indices or all")
    parser.add_argument("--render-percent", type=int, default=50)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.fps <= 0 or not 1 <= args.render_percent <= 100:
        raise ValueError("Invalid FPS or render resolution percentage")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene_path = root / "stage2_native_replay.blend"
    if scene_path == args.background_scene.resolve():
        raise ValueError("Never overwrite the original scene")
    with np.load(args.geometry, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    count = validate_geometry(data)
    if args.render_only:
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
        stored = json.loads(bpy.data.texts[PREFIX + "provenance.json"].as_string())
        if stored["geometry"] != binding(args.geometry):
            raise ValueError("Saved replay belongs to another geometry input")
    else:
        bpy.ops.wm.open_mainfile(filepath=str(args.background_scene.resolve()))
        if any(obj.name.startswith(PREFIX) for obj in bpy.data.objects):
            raise ValueError("Background already contains a replay; choose the original scene")
        for kind, color in (("human", (0.12, 0.48, 0.92, 1)),
                            ("object", (1.0, 0.28, 0.045, 1))):
            mesh_sequence(PREFIX + kind, data[kind + "_vertices"], data[kind + "_faces"], color)
        scene = bpy.context.scene
        height, width = map(int, data["image_size"])
        rotation, translation = data["camera_R"].reshape(3, 3), data["camera_T"].reshape(3)
        eye = -translation @ rotation.T
        source = camera(PREFIX + "source", eye, rotation @ np.diag([-1, 1, -1]),
                        float(data["focal"]), width)
        low = np.minimum(data["human_vertices"].min((0, 1)), data["object_vertices"].min((0, 1)))
        high = np.maximum(data["human_vertices"].max((0, 1)), data["object_vertices"].max((0, 1)))
        target = (low + high) / 2
        angle = np.deg2rad(55)
        yaw = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        novel_eye = target + yaw @ (eye - target)
        direction = Vector((target - novel_eye).tolist()).to_track_quat("-Z", "Y")
        camera(PREFIX + "oblique", novel_eye, np.asarray(direction.to_matrix()),
               0.85 * float(data["focal"]), width)
        scene.camera = source
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        scene.render.fps = round(args.fps)
        scene.render.fps_base = round(args.fps) / args.fps
        scene.frame_start, scene.frame_end = 1, count
        scene.frame_set(1)
        provenance = {"geometry": binding(args.geometry), "original_scene": binding(args.background_scene),
                      "script": binding(__file__), "frames": count, "fps": args.fps,
                      "model": args.model_label, "quality_status": args.quality_status,
                      "scope": "Unmodified native geometry replay; no fitting or contact repair",
                      "source_frame_to_blender_frame_offset": 1}
        block = bpy.data.texts.new(PREFIX + "provenance.json")
        block.write(json.dumps(provenance, indent=2))
        # A plain baked animation opens without executing embedded Python.
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene = bpy.context.scene
    scene.render.resolution_percentage = args.render_percent
    result = verify(data)
    result.update(geometry=binding(args.geometry), scene=binding(scene_path),
                  original_scene=binding(args.background_scene), model=args.model_label,
                  quality_status=args.quality_status, fps=args.fps,
                  no_pose_or_placement_edits=True, saved_scene_reopened=True)
    (root / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0.08, 0.08, 0.08)
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    frames = range(count) if args.frames == "all" else sorted({int(s) for s in args.frames.split(",")})
    for view in ("source", "oblique"):
        scene.camera = bpy.data.objects[PREFIX + view]
        folder = root / "renders" / view
        folder.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            if not 0 <= frame < count:
                raise ValueError("Requested source frame outside native sequence")
            scene.frame_set(frame + 1)
            scene.render.filepath = str(folder / f"{frame:05d}.png")
            bpy.ops.render.render(write_still=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
