"""Blender utilities: camera, masks, geometry, rendering, math.

Functions for GPU setup, camera positioning, mask rendering,
collision detection, quaternion math, and depth output.
"""

import glob
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import bpy  # isort: skip
import bmesh
import mathutils
import numpy as np
from mathutils.bvhtree import BVHTree

from grail.constants.image import FOCAL_LENGTH
from grail.core.io import save_init_rendering_data


def setup_gpu_rendering():
    """Configure Blender to use GPU for rendering"""
    print("Setting up GPU rendering...")

    # Get compute device type preference
    preferences = bpy.context.preferences
    cycles_prefs = preferences.addons["cycles"].preferences

    # Enable GPU compute
    cycles_prefs.compute_device_type = "CUDA"  # Try CUDA first

    # If CUDA not available, try OpenCL
    try:
        cycles_prefs.get_devices()
        if not any(device.type == "CUDA" for device in cycles_prefs.devices):
            print("CUDA not available, trying OpenCL...")
            cycles_prefs.compute_device_type = "OPENCL"
            cycles_prefs.get_devices()
    except:
        print("GPU setup failed, falling back to CPU")
        return False

    # Enable all available GPU devices
    gpu_found = False
    for device in cycles_prefs.devices:
        if device.type in {"CUDA", "OPENCL"}:
            device.use = True
            gpu_found = True
            print(f"Enabled GPU device: {device.name}")
        else:
            device.use = False  # Disable CPU when using GPU

    if not gpu_found:
        print("No GPU devices found, using CPU")
        return False

    # Set scene to use GPU
    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    return True


def configure_render_settings(output_path, samples=32, resolution_x=1920, resolution_y=1080):
    """Configure render settings for optimal quality and performance"""
    print("Configuring render settings...")

    scene = bpy.context.scene
    render = scene.render

    # Set render engine to Cycles for GPU support
    scene.render.engine = "CYCLES"

    # Configure output settings
    render.filepath = output_path
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.compression = 15

    # Set resolution
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = 100

    # Configure sampling
    scene.cycles.samples = samples

    # Enable denoising for better quality with fewer samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    # Optimize for GPU rendering
    scene.cycles.tile_size = 256

    print("Render settings configured:")
    print(f"  Resolution: {resolution_x}x{resolution_y}")
    print(f"  Samples: {samples}")
    print(f"  Output: {output_path}")


def load_blend_file(blend_file_path):
    """Load a .blend file"""
    if not os.path.exists(blend_file_path):
        raise FileNotFoundError(f"Blend file not found: {blend_file_path}")

    print(f"Loading blend file: {blend_file_path}")

    # Clear existing mesh objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Load the blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    print("Blend file loaded successfully")


def print_object_dimensions(obj):
    """
    Print the dimensions and bounding box information of a Blender object

    Args:
        obj: The Blender object
    """
    print("=" * 50)
    print("OBJECT DIMENSIONS")
    print("=" * 50)
    print(f"Object name: {obj.name}")

    # Get dimensions (already in world space considering scale)
    dimensions = obj.dimensions
    print(
        f"Dimensions (X, Y, Z): ({dimensions.x:.4f}, {dimensions.y:.4f}, {dimensions.z:.4f}) meters"
    )

    # Get bounding box corners
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # Calculate min and max coordinates
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)

    print("Bounding box:")
    print(f"  X: [{min_x:.4f}, {max_x:.4f}] (width: {max_x - min_x:.4f}m)")
    print(f"  Y: [{min_y:.4f}, {max_y:.4f}] (depth: {max_y - min_y:.4f}m)")
    print(f"  Z: [{min_z:.4f}, {max_z:.4f}] (height: {max_z - min_z:.4f}m)")

    # Get object center
    bbox_center = (
        sum((mathutils.Vector(corner) for corner in obj.bound_box), mathutils.Vector()) / 8
    )
    world_center = obj.matrix_world @ bbox_center
    print(
        f"Center (world space): ({world_center.x:.4f}, {world_center.y:.4f}, {world_center.z:.4f})"
    )
    print(f"Location: ({obj.location.x:.4f}, {obj.location.y:.4f}, {obj.location.z:.4f})")
    print(f"Scale: ({obj.scale.x:.4f}, {obj.scale.y:.4f}, {obj.scale.z:.4f})")
    print("=" * 50)


def get_blender_mesh_objects(obj):
    """
    Collect all MESH-type objects from a Blender object.
    Handles both direct MESH objects and parent empties (e.g. RenderPeople
    characters) that have MESH children.
    """
    if obj.type == "MESH":
        return [obj]
    return [child for child in obj.children_recursive if child.type == "MESH"]


def _bvh_world_space(obj, depsgraph):
    """Build a BVHTree for *obj* in world space (applies modifiers + world transform)."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mat = eval_obj.matrix_world
    verts = [mat @ v.co for v in mesh.vertices]
    polys = [tuple(p.vertices) for p in mesh.polygons]
    bvh = BVHTree.FromPolygons(verts, polys)
    eval_obj.to_mesh_clear()
    return bvh


def check_mesh_penetration(obj_a, obj_b):
    """
    Check whether two Blender objects have intersecting triangles using
    BVHTree.overlap().  Works with armature-deformed meshes because
    to_mesh() evaluates all modifiers (including Armature deformation).

    Both meshes are transformed into world space before comparison so that
    objects at different positions/rotations are correctly handled.

    Returns True if any triangle pair overlaps, False otherwise.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes_a = get_blender_mesh_objects(obj_a)
    meshes_b = get_blender_mesh_objects(obj_b)

    for ma in meshes_a:
        bvh_a = _bvh_world_space(ma, depsgraph)
        for mb in meshes_b:
            bvh_b = _bvh_world_space(mb, depsgraph)
            if bvh_a.overlap(bvh_b):
                return True
    return False


def check_character_penetration(character, static_objects, loaded_obj):
    """
    Check if *character* penetrates any of the *static_objects* or the
    *loaded_obj*.

    Returns True when penetration is detected, False otherwise.
    """
    for obj_key, obj in static_objects.items():
        if check_mesh_penetration(character, obj):
            print(f"Penetration detected: character vs static object '{obj_key}'")
            return True

    if loaded_obj is not None:
        if check_mesh_penetration(character, loaded_obj):
            print("Penetration detected: character vs loaded object")
            return True

    return False


def sample_xy_in_box(include_box, exclude_box=None, max_retries=1000):
    """
    Sample a (dx, dy) uniformly within *include_box* but outside *exclude_box*.

    Args:
        include_box: [x1, y1, x2, y2] bounding box to sample within.
        exclude_box: optional [x1, y1, x2, y2] region to reject.
        max_retries: rejection-sampling budget.

    Returns:
        (dx, dy) tuple.
    """
    x1_inc, y1_inc, x2_inc, y2_inc = include_box
    for _ in range(max_retries):
        dx = random.uniform(x1_inc, x2_inc)
        dy = random.uniform(y1_inc, y2_inc)
        if exclude_box is not None:
            x1_exc, y1_exc, x2_exc, y2_exc = exclude_box
            if x1_exc <= dx <= x2_exc and y1_exc <= dy <= y2_exc:
                continue
        return dx, dy
    print(
        f"Warning: could not sample outside exclude box after {max_retries} retries, returning last sample"
    )
    return dx, dy


def position_camera_around_object(
    camera,
    target_object,
    target_character,
    elevation_deg,
    azimuth_deg,
    radius,
    camera_offset=(0.0, 0.0, 0.0),
):
    """
    Position camera to look at target object with spherical coordinates
    Similar to the approach in render_objects.py

    Args:
        camera: Blender camera object
        target_object: Object to look at (center target)
        elevation_deg: Elevation angle in degrees
        azimuth_deg: Azimuth angle in degrees
        radius: Distance from target
        camera_offset: Additional (x, y, z) offset for camera position
    """
    import numpy as np

    # Convert angles to radians
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)

    # Get target object center (bounding box center)
    target_location = target_object.location
    if target_character:
        target_location = (target_object.location + target_character.location) / 2
        target_location[2] = target_object.location[2]

    # Calculate camera position using spherical coordinates
    # Similar to render_objects.py approach
    camera_x = radius * math.cos(elevation) * math.cos(azimuth) + camera_offset[0]
    camera_y = radius * math.cos(elevation) * math.sin(azimuth) + camera_offset[1]
    camera_z = radius * math.sin(elevation) + camera_offset[2]

    # Position relative to target object
    camera_location = (
        target_location.x + camera_x,
        target_location.y + camera_y,
        target_location.z + camera_z,
    )

    # Clear any parent relationship to ensure world coordinates
    if camera.parent:
        print(f"Clearing camera parent relationship with: {camera.parent.name}")
        camera.parent = None
        camera.parent_type = "OBJECT"

    # Set camera location in world coordinates
    camera.location = camera_location

    # Calculate rotation to look at target
    # Similar to render_objects.py rotation calculation
    camera_rotation = (math.pi / 2 - elevation, 0, math.pi / 2 + azimuth)

    camera.rotation_euler = camera_rotation

    # Force update the scene and matrix_world
    bpy.context.view_layer.update()

    print("Camera positioned around object:")
    print(f"  Target object: {target_object.name} at {target_location}")
    print(f"  Elevation: {elevation_deg}°, Azimuth: {azimuth_deg}°, Radius: {radius}")
    print(
        f"  Camera position: ({camera.location.x:.3f}, {camera.location.y:.3f}, {camera.location.z:.3f})"
    )
    print(
        f"  Camera rotation: ({math.degrees(camera.rotation_euler.x):.1f}°, {math.degrees(camera.rotation_euler.y):.1f}°, {math.degrees(camera.rotation_euler.z):.1f}°)"
    )


def compute_polar_character_placement(scene_config, obj_pos, rand_seed=None, set_seed=True):
    """
    Compute character position and rotation using polar coordinates relative to the object.

    Args:
        scene_config: Scene configuration dict with character_azimuth, character_distance, etc.
        obj_pos: Object position [x, y, z]
        rand_seed: Random seed for azimuth/distance randomization (None = no randomization)
        set_seed: If True, call random.seed() for deterministic results; if False, skip seeding

    Returns:
        (character_position, character_rotation) as lists of floats (rotation in radians)
    """
    azimuth_deg = scene_config["character_azimuth"]
    distance = scene_config["character_distance"]

    if rand_seed is not None:
        if set_seed:
            random.seed(rand_seed)
        if "character_azimuth_rand_max" in scene_config:
            azimuth_deg += random.uniform(
                -scene_config["character_azimuth_rand_max"],
                scene_config["character_azimuth_rand_max"],
            )
        if "character_distance_rand_max" in scene_config:
            distance += random.uniform(
                -scene_config["character_distance_rand_max"],
                scene_config["character_distance_rand_max"],
            )
            distance = max(distance, 0.1)

    azimuth_rad = math.radians(azimuth_deg)
    char_x = obj_pos[0] + distance * math.cos(azimuth_rad)
    char_y = obj_pos[1] + distance * math.sin(azimuth_rad)
    char_z = scene_config.get("character_z", obj_pos[2])

    character_position = [char_x, char_y, char_z]

    # RenderPeople at [90, 0, 0] faces -Y; offset = +pi/2
    x_rot = math.radians(90)
    if scene_config.get("character_face_object", False):
        dx = obj_pos[0] - char_x
        dy = obj_pos[1] - char_y
        face_angle = math.atan2(dy, dx)
        z_rot = face_angle + math.pi / 2
    else:
        z_rot = math.radians(scene_config.get("character_rot", [90, 0, 0])[2])

    character_rotation = [x_rot, 0.0, z_rot]

    print("Polar character placement:")
    print(f"  Azimuth: {azimuth_deg:.1f}°, Distance: {distance:.2f}m")
    print(f"  Position: ({char_x:.3f}, {char_y:.3f}, {char_z:.4f})")
    print(f"  Rotation: ({math.degrees(x_rot):.1f}°, 0.0°, {math.degrees(z_rot):.1f}°)")

    return character_position, character_rotation


def setup_camera(
    width,
    height,
    senser_size=10,
    target_object=None,
    target_character=None,
    elevation=None,
    azimuth=None,
    radius=None,
    camera_offset=None,
):
    """Ensure camera is properly configured with focal length from config"""
    scene = bpy.context.scene

    # Find camera in scene
    camera = None
    for obj in scene.objects:
        if obj.type == "CAMERA":
            camera = obj
            break

    if camera is None:
        print("No camera found in scene, creating default camera")
        bpy.ops.object.camera_add(location=(7.35, -6.93, 4.96))
        camera = bpy.context.object
        camera.rotation_euler = (1.1, 0, 0.814)

    # Set as active camera
    scene.camera = camera

    # Configure camera with focal length from config

    camera_data = camera.data
    if camera_data.sensor_fit != "AUTO":
        raise ValueError("sensor_fit must be AUTO")

    # WARNING: AUTO mode will apply sensor_width to larger dimension
    if width > height:
        focal_length_mm = FOCAL_LENGTH * senser_size / width
    else:
        focal_length_mm = FOCAL_LENGTH * senser_size / height
    camera_data.lens = focal_length_mm  # Set focal length in mm
    camera_data.sensor_width = senser_size
    camera_data.type = "PERSP"
    camera_data.clip_start = 1
    camera_data.clip_end = 6000

    # Position camera around target object if specified
    if target_object and elevation is not None and azimuth is not None and radius is not None:
        position_camera_around_object(
            camera,
            target_object,
            target_character,
            elevation,
            azimuth,
            radius,
            camera_offset or (0.0, 0.0, 0.0),
        )

    # Ensure scene is updated for consistent matrix_world
    bpy.context.view_layer.update()

    print(f"Using camera: {camera.name}")
    print(f"Camera focal length set to: {FOCAL_LENGTH:.2f}mm")
    print(
        f"Camera position: ({camera.location.x:.3f}, {camera.location.y:.3f}, {camera.location.z:.3f})"
    )
    print(
        f"Camera rotation (euler): ({camera.rotation_euler.x:.3f}, {camera.rotation_euler.y:.3f}, {camera.rotation_euler.z:.3f}) radians"
    )
    print(
        f"Camera rotation (degrees): ({math.degrees(camera.rotation_euler.x):.1f}°, {math.degrees(camera.rotation_euler.y):.1f}°, {math.degrees(camera.rotation_euler.z):.1f}°)"
    )

    return camera


def save_camera_and_object_data(camera, obj, camera_save_path, focal_length, resolution):
    """
    Save camera and object data to pickle file, similar to render_objects.py

    Args:
        camera: Blender camera object
        obj: Blender object being rendered
        camera_save_path (str): Path to save camera data
        focal_length (float): Camera focal length
        resolution (tuple): Render resolution
    """
    # Ensure scene is updated
    bpy.context.view_layer.update()

    camera_matrix_world = camera.matrix_world
    R = np.array(camera_matrix_world)[:3, :3]  # 3 x 3
    t = np.array(camera_matrix_world)[:3, 3]  # 3 x 1

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(camera_save_path), exist_ok=True)

    camera_data = dict(
        R=R,
        t=t,
        obj_R=np.array(obj.rotation_euler.to_matrix()).reshape((3, 3)),  # 3 x 3
        obj_euler=np.array(obj.rotation_euler).reshape((3, 1)),  # 3 x 1
        obj_location=np.array(obj.location).reshape((3, 1)),  # 3 x 1
        obj_t=np.array(obj.location).reshape((3, 1)),  # 3 x 1
        obj_scale=np.array(obj.scale).reshape((3, 1)),  # 3 x 1
        focal_length=focal_length,
        resolution=resolution,
    )

    with open(camera_save_path, "wb") as handle:
        pickle.dump(camera_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Camera and object data saved to: {camera_save_path}")


def save_foundation_pose_camera_data(
    camera,
    obj,
    character,
    static_objects,
    frame_height,
    frame_width,
    focal_length,
    foundation_pose_camera_save_path,
    support_object_names=None,
):
    """
    Save camera and object data to pickle file in FoundationPose format
    Similar to render_blender_scene_fp.py format with cam_R, cam_t, obj_R, obj_t

    Args:
        camera: Blender camera object
        obj: Blender object being rendered
        foundation_pose_camera_save_path (str): Path to save FoundationPose camera data
    """
    # Ensure scene is updated
    bpy.context.view_layer.update()

    camera_matrix_world = camera.matrix_world
    cam_R = np.array(camera_matrix_world)[:3, :3]  # 3 x 3
    cam_t = np.array(camera_matrix_world)[:3, 3]  # 3 x 1

    # Built-in support meshes are read only here, outside the placement/randomization loop.
    if support_object_names is None:
        support_object_names = []
    if (
        not isinstance(support_object_names, (list, tuple))
        or any(not isinstance(name, str) or not name for name in support_object_names)
        or len(set(support_object_names)) != len(support_object_names)
    ):
        raise ValueError("support_object_names must contain unique exact Blender object names")
    support_objects = {}
    for name in support_object_names:
        support = bpy.context.scene.objects.get(name)
        if support is None:
            raise ValueError(f"Selected support object {name!r} is missing from the scene")
        support_objects[name] = support

    save_init_rendering_data(
        foundation_pose_camera_save_path,
        cam_R,
        cam_t,
        frame_height,
        frame_width,
        focal_length,
        obj.rotation_euler.to_matrix(),
        obj.location,
        obj.scale,
        character.rotation_euler.to_matrix(),
        character.location,
        character.scale,
        static_objects,
        support_objects=support_objects,
        source_blend_path=bpy.data.filepath,
    )

    # # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(foundation_pose_camera_save_path), exist_ok=True)

    # # FoundationPose format - simplified data structure
    # foundation_pose_data = dict(
    #     cam_R=cam_R,
    #     cam_t=cam_t,
    #     obj_R=np.array(obj.rotation_euler.to_matrix()).reshape((3, 3)),  # 3 x 3
    #     obj_t=np.array(obj.location).reshape((3, 1)),  # 3 x 1
    #     obj_scale=np.array(obj.scale).reshape((3, 1)),  # 3 x 1
    # )

    # with open(foundation_pose_camera_save_path, "wb") as handle:
    #     pickle.dump(foundation_pose_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FoundationPose camera data saved to: {foundation_pose_camera_save_path}")


def load_obj_scale_data(dataset, category, obj_scale_dir):
    """
    Load object scale data from scale determination step.

    Args:
        dataset (str): Dataset name
        category (str): Object category
        obj_scale_dir (str): Directory containing obj_scale JSON files

    Returns:
        list or None: Object scale [x, y, z] if found, None otherwise
    """
    import json as _json

    obj_scale_file = f"{obj_scale_dir}/{dataset}/{category}/obj_scale.json"

    if not os.path.exists(obj_scale_file):
        print(f"Object scale file not found: {obj_scale_file}")
        return None

    try:
        with open(obj_scale_file, "r") as f:
            data = _json.load(f)

        obj_scale = data["obj_scale"]
        print(f"Loaded object scale from: {obj_scale_file}")
        print(f"  obj_scale: {obj_scale}")
        return obj_scale
    except Exception as e:
        print(f"Failed to load object scale data: {str(e)}")
        return None


def load_initial_state_data(dataset, category, initial_state_dir):
    """
    Load initial state data from simulation

    Args:
        dataset (str): Dataset name
        category (str): Object category
        initial_state_dir (str): Directory containing initial state files

    Returns:
        dict or None: Initial state data if found, None otherwise
    """
    initial_state_file = f"{initial_state_dir}/{dataset}/{category}/initial_state.pickle"

    if not os.path.exists(initial_state_file):
        print(f"Initial state file not found: {initial_state_file}")
        return None

    try:
        with open(initial_state_file, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded initial state data from: {initial_state_file}")
        return data
    except Exception as e:
        print(f"Failed to load initial state data: {str(e)}")
        return None


def quaternion_from_euler(euler_angles):
    """
    Convert Euler angles to quaternion

    Args:
        euler_angles (array): Euler angles [x, y, z] in degrees

    Returns:
        array: Quaternion [x, y, z, w]
    """
    # Convert degrees to radians
    roll, pitch, yaw = [math.radians(angle) for angle in euler_angles]

    # Calculate quaternion components
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # Quaternion multiplication for XYZ order
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions

    Args:
        q1 (array): First quaternion [x, y, z, w]
        q2 (array): Second quaternion [x, y, z, w]

    Returns:
        array: Result quaternion [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def quaternion_to_euler_blender(quaternion, post_rotation=None):
    """
    Convert quaternion to Blender euler angles (XYZ order)

    Args:
        quaternion (array): Quaternion [x, y, z, w] from Warp simulation
        post_rotation (array): Post-rotation euler angles (x, y, z) in degrees

    Returns:
        tuple: Euler angles (x, y, z) in radians for Blender
    """
    if post_rotation is not None:
        quaternion = quaternion_multiply(quaternion_from_euler(post_rotation), quaternion)

    x, y, z, w = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def get_camera_intrinsics(camera, render_width, render_height):
    """
    Calculate camera intrinsic parameters

    Args:
        camera: Blender camera object
        render_width: Render resolution width
        render_height: Render resolution height

    Returns:
        dict: Dictionary containing intrinsic parameters
    """
    camera_data = camera.data

    # Get sensor dimensions
    # sensor_width = camera_data.sensor_width  # in mm
    if render_width > render_height:
        sensor_width = camera_data.sensor_width
        sensor_height = sensor_width * (render_height / render_width)
    else:
        sensor_height = camera_data.sensor_width
        sensor_width = sensor_height * (render_width / render_height)
    # sensor_height = camera_data.sensor_height  # in mm

    # print(f"sensor_width: {sensor_width}, sensor_height: {sensor_height}")

    # # Calculate sensor height based on aspect ratio if sensor_fit is AUTO
    # if camera_data.sensor_fit == "AUTO":
    #     if render_width >= render_height:
    #         # Landscape: sensor_width is the reference
    #         sensor_height = sensor_width * (render_height / render_width)
    #     else:
    #         # Portrait: sensor_height is the reference
    #         sensor_width = sensor_height * (render_width / render_height)
    # elif camera_data.sensor_fit == "HORIZONTAL":
    #     sensor_height = sensor_width * (render_height / render_width)
    # elif camera_data.sensor_fit == "VERTICAL":
    #     sensor_width = sensor_height * (render_width / render_height)

    # Focal length in mm
    focal_length_mm = camera_data.lens

    # Calculate focal length in pixels
    fx = (focal_length_mm / sensor_width) * render_width
    fy = (focal_length_mm / sensor_height) * render_height

    # Principal point (optical center) - usually at image center
    cx = render_width / 2.0
    cy = render_height / 2.0

    # Intrinsic matrix K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    intrinsics = {
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": sensor_width,
        "sensor_height_mm": sensor_height,
        "render_width": render_width,
        "render_height": render_height,
        "sensor_fit": camera_data.sensor_fit,
    }

    return intrinsics


def print_camera_intrinsics(intrinsics):
    """Print camera intrinsic parameters in a formatted way"""
    print("=" * 50)
    print("CAMERA INTRINSIC PARAMETERS")
    print("=" * 50)
    print(f"Focal length (mm): {intrinsics['focal_length_mm']:.3f}")
    print(
        f"Sensor size (mm): {intrinsics['sensor_width_mm']:.3f} x {intrinsics['sensor_height_mm']:.3f}"
    )
    print(f"Image resolution: {intrinsics['render_width']} x {intrinsics['render_height']}")
    print(f"Sensor fit: {intrinsics['sensor_fit']}")
    print(f"Focal length (pixels): fx={intrinsics['fx']:.3f}, fy={intrinsics['fy']:.3f}")
    print(f"Principal point: cx={intrinsics['cx']:.3f}, cy={intrinsics['cy']:.3f}")
    print("Intrinsic matrix K:")
    print(
        f"  [{intrinsics['K'][0, 0]:8.3f} {intrinsics['K'][0, 1]:8.3f} {intrinsics['K'][0, 2]:8.3f}]"
    )
    print(
        f"  [{intrinsics['K'][1, 0]:8.3f} {intrinsics['K'][1, 1]:8.3f} {intrinsics['K'][1, 2]:8.3f}]"
    )
    print(
        f"  [{intrinsics['K'][2, 0]:8.3f} {intrinsics['K'][2, 1]:8.3f} {intrinsics['K'][2, 2]:8.3f}]"
    )
    print("=" * 50)


# def save_intrinsics_to_file(intrinsics, output_path):
#     """
#     Save camera intrinsic matrix to file in the same format as cam_K.txt

#     Args:
#         intrinsics: Dictionary containing intrinsic parameters
#         output_path: Path to save the cam_K.txt file
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     K = intrinsics["K"]

#     # Format: each element with scientific notation and 18 decimal places
#     # Same format as the existing cam_K.txt file
#     with open(output_path, "w") as f:
#         # Row 1: fx, 0, cx
#         f.write(f"{K[0, 0]:.18e} {K[0, 1]:.18e} {K[0, 2]:.18e}\n")
#         # Row 2: 0, fy, cy
#         f.write(f"{K[1, 0]:.18e} {K[1, 1]:.18e} {K[1, 2]:.18e}\n")
#         # Row 3: 0, 0, 1
#         f.write(f"{K[2, 0]:.18e} {K[2, 1]:.18e} {K[2, 2]:.18e}\n")

#     print(f"Camera intrinsics saved to: {output_path}")


def setup_object_mask_material(obj, mask_color=(1, 1, 1, 1)):
    """
    Create a pure color material for object masking

    Args:
        obj: Blender object to apply mask material to (can be a mesh or parent with mesh children)
        mask_color: RGBA color for the mask (default: white)
    """
    meshes = get_blender_mesh_objects(obj)
    if meshes and all(
        mesh.data.materials
        and all(
            material and material.get("source_material", "").startswith("/DHGen_Root/")
            for material in mesh.data.materials
        )
        for mesh in meshes
    ):
        from grail.pipelines.blender.dhgen import mask_material

        for mesh in meshes:
            for index, material in enumerate(list(mesh.data.materials)):
                mesh.data.materials[index] = mask_material(material, mask_color)
        return
    # Create mask material
    mask_mat = bpy.data.materials.new(name=f"{obj.name}_Mask")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    links = mask_mat.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add Emission shader for pure color output
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = mask_color
    emission.inputs["Strength"].default_value = 1.0
    emission.location = (0, 0)

    # Add Material Output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)

    # Connect emission to output
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # Apply material to mesh object(s)
    # Check if this is a mesh object or a parent with mesh children
    target_meshes = []
    if obj.type == "MESH":
        target_meshes.append(obj)
    elif hasattr(obj, "children"):
        # If it's a parent object, apply to all mesh children
        for child in obj.children_recursive:
            if child.type == "MESH":
                target_meshes.append(child)

    # Replace all materials with mask material
    for mesh_obj in target_meshes:
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(mask_mat)
        print(f"Applied mask material to: {mesh_obj.name}")


def render_object_mask(obj, mask_output_path, background_color=(0, 0, 0)):
    """
    Render a binary mask of the object

    Args:
        obj: Blender object to create mask for
        mask_output_path: Path to save the mask image
        background_color: RGB background color (default: black)
    """
    print(f"Rendering object mask for: {obj.name}")

    scene = bpy.context.scene

    # Store original settings
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    original_filepath = scene.render.filepath

    # Configure render settings for mask
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.filepath = mask_output_path

    # Set world background to black
    world = scene.world
    if world and world.use_nodes:
        world_nodes = world.node_tree.nodes
        for node in world_nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (*background_color, 1.0)
                node.inputs["Strength"].default_value = 1.0

    # Store original materials for all objects
    original_materials = {}
    original_material_indices = {}
    for scene_obj in scene.objects:
        if scene_obj.type == "MESH":
            original_materials[scene_obj.name] = scene_obj.data.materials[:]
            original_material_indices[scene_obj.name] = [
                polygon.material_index for polygon in scene_obj.data.polygons
            ]

    # Hide all objects except the target object (and its children if it's a parent)
    original_visibility = {}
    target_objects = set()

    # Collect target objects (the object itself and its mesh children if it's a parent)
    if obj.type == "MESH":
        target_objects.add(obj)
    elif hasattr(obj, "children"):
        for child in obj.children_recursive:
            if child.type == "MESH":
                target_objects.add(child)

    for scene_obj in scene.objects:
        original_visibility[scene_obj.name] = scene_obj.hide_render
        if scene_obj not in target_objects and scene_obj.type == "MESH":
            scene_obj.hide_render = True

    # Apply white mask material to target object
    setup_object_mask_material(obj, mask_color=(1, 1, 1, 1))

    # Render the mask
    print("Rendering mask...")
    start_time = time.time()
    bpy.ops.render.render(write_still=True)
    end_time = time.time()

    print(f"Mask rendered in {end_time - start_time:.2f} seconds")
    print(f"Mask saved to: {mask_output_path}")

    # Import cv2 here to avoid dependency issues if not needed
    try:
        import cv2

        # Convert RGBA to single channel mask (use alpha channel)
        mask_image = cv2.imread(mask_output_path, cv2.IMREAD_UNCHANGED)
        if mask_image.shape[2] == 4:  # RGBA
            mask_image = mask_image[:, :, 3]  # Extract alpha channel
        else:  # RGB - convert to grayscale
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(mask_output_path, mask_image)
    except ImportError:
        print("Warning: OpenCV not available, mask saved as RGBA PNG")

    # Restore original settings
    scene.render.engine = original_engine
    scene.render.film_transparent = original_film_transparent
    scene.render.filepath = original_filepath

    # Restore original visibility
    for scene_obj in scene.objects:
        if scene_obj.name in original_visibility:
            scene_obj.hide_render = original_visibility[scene_obj.name]

    # Restore original materials
    for scene_obj in scene.objects:
        if scene_obj.type == "MESH" and scene_obj.name in original_materials:
            scene_obj.data.materials.clear()
            for mat in original_materials[scene_obj.name]:
                scene_obj.data.materials.append(mat)
            for polygon, index in zip(
                scene_obj.data.polygons, original_material_indices[scene_obj.name]
            ):
                polygon.material_index = index


def setup_holdout_material(obj):
    """
    Create a holdout material that blocks/occludes other objects without contributing color

    Args:
        obj: Blender object to apply holdout material to
    """
    # Create holdout material
    holdout_mat = bpy.data.materials.new(name=f"{obj.name}_Holdout")
    holdout_mat.use_nodes = True
    nodes = holdout_mat.node_tree.nodes
    links = holdout_mat.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add Holdout shader - this makes the object invisible but still occludes
    holdout = nodes.new(type="ShaderNodeHoldout")
    holdout.location = (0, 0)

    # Add Material Output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)

    # Connect holdout to output
    links.new(holdout.outputs["Holdout"], output.inputs["Surface"])

    # Apply material to mesh object
    if obj.type == "MESH":
        obj.data.materials.clear()
        obj.data.materials.append(holdout_mat)


def render_character_mask(
    parent_obj, mask_output_path, background_color=(0, 0, 0), consider_occlusion=True
):
    """
    Render a binary mask of a character (parent object with mesh children)

    Args:
        parent_obj: Parent object containing the character meshes
        mask_output_path: Path to save the mask image
        background_color: RGB background color (default: black)
        consider_occlusion: If True, other objects will occlude the character in the mask
    """
    print(f"Rendering character mask for: {parent_obj.name}")
    print(f"Consider occlusion: {consider_occlusion}")

    scene = bpy.context.scene

    # Store original settings
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    original_filepath = scene.render.filepath

    # Configure render settings for mask
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.filepath = mask_output_path

    # Set world background to black
    world = scene.world
    if world and world.use_nodes:
        world_nodes = world.node_tree.nodes
        for node in world_nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (*background_color, 1.0)
                node.inputs["Strength"].default_value = 1.0

    # Store original materials for all objects
    original_materials = {}
    original_material_indices = {}
    for scene_obj in scene.objects:
        if scene_obj.type == "MESH":
            original_materials[scene_obj.name] = scene_obj.data.materials[:]
            original_material_indices[scene_obj.name] = [
                polygon.material_index for polygon in scene_obj.data.polygons
            ]

    # Get all mesh children of the parent object
    character_meshes = []
    if parent_obj.type == "MESH":
        character_meshes.append(parent_obj)
    for child in parent_obj.children_recursive:
        if child.type == "MESH":
            character_meshes.append(child)

    print(f"Found {len(character_meshes)} mesh object(s) in character")

    original_visibility = {}
    if consider_occlusion:
        # Keep all objects visible but apply holdout material to non-character objects
        # This makes them occlude the character without contributing to the mask
        for scene_obj in scene.objects:
            original_visibility[scene_obj.name] = scene_obj.hide_render
            if scene_obj not in character_meshes and scene_obj.type == "MESH":
                setup_holdout_material(scene_obj)
    else:
        # Original behavior: hide all objects except the character meshes
        for scene_obj in scene.objects:
            original_visibility[scene_obj.name] = scene_obj.hide_render
            if scene_obj not in character_meshes and scene_obj.type == "MESH":
                scene_obj.hide_render = True

    # Apply white mask material to all character meshes
    for mesh_obj in character_meshes:
        setup_object_mask_material(mesh_obj, mask_color=(1, 1, 1, 1))

    # Render the mask
    print("Rendering character mask...")
    start_time = time.time()
    bpy.ops.render.render(write_still=True)
    end_time = time.time()

    print(f"Character mask rendered in {end_time - start_time:.2f} seconds")
    print(f"Character mask saved to: {mask_output_path}")

    # Import cv2 here to avoid dependency issues if not needed
    try:
        import cv2

        # Convert RGBA to single channel mask (use alpha channel)
        mask_image = cv2.imread(mask_output_path, cv2.IMREAD_UNCHANGED)
        if mask_image.shape[2] == 4:  # RGBA
            mask_image = mask_image[:, :, 3]  # Extract alpha channel
        else:  # RGB - convert to grayscale
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(mask_output_path, mask_image)
    except ImportError:
        print("Warning: OpenCV not available, mask saved as RGBA PNG")

    # Restore original settings
    scene.render.engine = original_engine
    scene.render.film_transparent = original_film_transparent
    scene.render.filepath = original_filepath

    # Restore original visibility
    for scene_obj in scene.objects:
        if scene_obj.name in original_visibility:
            scene_obj.hide_render = original_visibility[scene_obj.name]

    # Restore original materials
    for scene_obj in scene.objects:
        if scene_obj.type == "MESH" and scene_obj.name in original_materials:
            scene_obj.data.materials.clear()
            for mat in original_materials[scene_obj.name]:
                scene_obj.data.materials.append(mat)
            for polygon, index in zip(
                scene_obj.data.polygons, original_material_indices[scene_obj.name]
            ):
                polygon.material_index = index


def setup_depth_output(depth_output_path):
    """
    Set up compositor nodes to output depth map

    Args:
        depth_output_path (str): Path to save the depth map (without extension)

    Returns:
        The file output node for depth
    """
    scene = bpy.context.scene

    # Enable use of compositor nodes
    scene.use_nodes = True

    # Enable Z pass in render layers
    scene.view_layers["ViewLayer"].use_pass_z = True

    # Get the compositor node tree
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Find or create Render Layers node
    render_layers_node = None
    for node in nodes:
        if node.type == "R_LAYERS":
            render_layers_node = node
            break

    if render_layers_node is None:
        render_layers_node = nodes.new(type="CompositorNodeRLayers")
        render_layers_node.location = (0, 0)

    # Create File Output node for depth
    depth_output_node = nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.name = "Depth Output"
    depth_output_node.label = "Depth Output"
    depth_output_node.location = (600, -200)

    # Configure depth output
    depth_output_node.base_path = os.path.dirname(depth_output_path)
    depth_output_node.file_slots[0].path = os.path.basename(depth_output_path)

    # Use OpenEXR format for accurate depth values
    depth_output_node.format.file_format = "OPEN_EXR"
    depth_output_node.format.color_depth = "32"
    depth_output_node.format.exr_codec = "ZIP"

    # Connect Depth output from Render Layers to File Output
    links.new(render_layers_node.outputs["Depth"], depth_output_node.inputs[0])

    print(f"Depth output configured: {depth_output_path}")

    return depth_output_node


def save_depth_as_png(exr_path, png_path, near_clip=0.1, far_clip=100.0):
    """
    Convert EXR depth map to 16-bit PNG with millimeter scale (metric depth)

    Args:
        exr_path (str): Path to the EXR depth file
        png_path (str): Path to save the PNG depth (16-bit, millimeter scale)
        near_clip (float): Near clipping distance (meters)
        far_clip (float): Far clipping distance (meters)

    Note:
        Output is uint16 PNG where pixel value = depth in millimeters
        Valid range: 0-65535 mm (0-65.535 meters)
        Pixels beyond far_clip are clamped to 65535
    """
    depth = None

    # Method 1: Try OpenCV with EXR support enabled
    try:
        import cv2

        # Enable OpenEXR support (must be set before imread)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        depth = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            print("Read EXR using OpenCV")
    except Exception as e:
        print(f"OpenCV EXR read failed: {e}")

    # Method 2: Try imageio (supports EXR via freeimage or OpenEXR backend)
    if depth is None:
        try:
            import imageio.v3 as iio

            depth = iio.imread(exr_path)
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            print("Read EXR using imageio")
        except Exception as e:
            print(f"imageio EXR read failed: {e}")

    # Method 3: Try OpenEXR library directly
    if depth is None:
        try:
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = list(header["channels"].keys())
            # Try common depth channel names
            for ch_name in ["Z", "R", "View Layer.Depth.Z"] + channels:
                if ch_name in channels:
                    depth_str = exr_file.channel(ch_name, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
                    print(f"Read EXR using OpenEXR library (channel: {ch_name})")
                    break
        except Exception as e:
            print(f"OpenEXR library read failed: {e}")

    if depth is None:
        print(f"Warning: Could not read depth EXR: {exr_path}")
        print("Install one of: imageio[freeimage], OpenEXR, or rebuild opencv with EXR support")
        return False

    # Convert from meters to millimeters
    depth_mm = depth * 1000.0

    # Clip to uint16 range (0-65535 mm = 0-65.535 meters)
    depth_mm = np.clip(depth_mm, 0, 65535)

    # Convert to uint16
    depth_uint16 = depth_mm.astype(np.uint16)

    # Save PNG
    try:
        import cv2

        cv2.imwrite(png_path, depth_uint16)
    except:
        try:
            import imageio.v3 as iio

            iio.imwrite(png_path, depth_uint16)
        except:
            from PIL import Image

            Image.fromarray(depth_uint16).save(png_path)

    print(f"Depth PNG saved to: {png_path} (16-bit, millimeter scale)")
    return True


def render_scene(depth_output_path=None):
    """Render the current scene

    Args:
        depth_output_path (str, optional): Path to save depth map (without extension)
    """
    print("Starting render...")
    start_time = time.time()

    # Setup depth output if requested
    depth_output_node = None
    if depth_output_path:
        depth_output_node = setup_depth_output(depth_output_path)

    # Render
    bpy.ops.render.render(write_still=True)

    end_time = time.time()
    render_time = end_time - start_time
    print(f"Render completed in {render_time:.2f} seconds")

    # Convert depth EXR to PNG for visualization
    if depth_output_path:
        # Blender adds frame number to the output, find the actual file
        exr_files = glob.glob(f"{depth_output_path}*.exr")
        if exr_files:
            exr_path = exr_files[0]
            png_path = depth_output_path + ".png"

            # Get camera clip distances for proper normalization
            scene = bpy.context.scene
            if scene.camera:
                near_clip = scene.camera.data.clip_start
                far_clip = scene.camera.data.clip_end
            else:
                near_clip = 0.1
                far_clip = 100.0

            save_depth_as_png(exr_path, png_path, near_clip, far_clip)

        # Clean up the depth output node
        if depth_output_node:
            scene = bpy.context.scene
            scene.node_tree.nodes.remove(depth_output_node)


def save_character_data(character, character_name, save_path):
    """Save character metadata (height, position, scale) to a pickle file.

    Args:
        character: Blender object for the character
        character_name: Character identifier string
        save_path: Path to save the character_data.pickle
    """
    import pickle

    # Compute character height from world-space bounding box of all mesh children
    all_z = []
    for child in character.children_recursive:
        if child.type == "MESH":
            for corner in child.bound_box:
                world_corner = child.matrix_world @ mathutils.Vector(corner)
                all_z.append(world_corner.z)
    # Fallback to parent object if no mesh children
    if not all_z:
        for corner in character.bound_box:
            world_corner = character.matrix_world @ mathutils.Vector(corner)
            all_z.append(world_corner.z)
    character_height = max(all_z) - min(all_z) if all_z else 0.0

    # Extract transform
    character_R = np.array(character.matrix_world.to_3x3())
    character_t = np.array(character.matrix_world.translation)
    character_scale = np.array(character.scale)

    data = {
        "character_name": character_name,
        "character_height": float(character_height),
        "character_scale": character_scale,
        "character_R": character_R,
        "character_t": character_t,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Character data saved to: {save_path}")
    print(f"  Character height: {character_height:.4f}m")


def rand_range_3d(cfg, key_max, key_min=None):
    """Build per-axis [lo, hi] ranges from config, defaulting lo to -max."""
    max_vals = cfg[key_max]
    min_vals = cfg.get(key_min) if key_min else None
    return [[min_vals[i] if min_vals else -max_vals[i], max_vals[i]] for i in range(3)]


def sample_rand_3d(cfg, key_max, key_min=None):
    """Sample a random 3D offset from config range."""
    rng = rand_range_3d(cfg, key_max, key_min)
    return [random.uniform(rng[i][0], rng[i][1]) for i in range(3)]
