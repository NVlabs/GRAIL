"""Blender asset loading: objects, characters, textures.

Functions for loading 3D objects (OBJ/GLB), rigged characters, G1 robots,
static scene objects, and managing textures.
"""

import glob
import hashlib
import math
import os
import random
import shutil
from pathlib import Path

import bpy  # isort: skip
import bmesh
import mathutils
import numpy as np

from grail.core.dataset import category2object


def load_object_from_category(dataset, category, position, rotation, scale):
    """
    Load object by globbing for meshes under data/<dataset>/<category>.

    Args:
        dataset (str): Dataset name (folder under data/ containing category subfolders).
        category (str): Object category
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians
        scale (tuple): (x, y, z) scale factors

    Returns:
        object: The imported Blender object
    """
    # Get object path from category
    object_path = category2object(f"data/{dataset}", category)
    print(f"Loading object: {object_path}")

    if dataset == "SAM3D":
        return load_glb_asset(object_path.replace(".obj", ".glb"), position, rotation, scale)
    else:
        return load_obj_asset(object_path, position, rotation, scale)


def repose_renderpeople(parent_obj, rand_seed=None):
    """
    Change the pose of a RenderPeople character to a relaxed pose, and randomize the pose if rand_seed is provided

    Args:
        parent_obj: The parent object containing the RenderPeople character
        rand_seed: The seed to randomize the pose
    """
    if parent_obj.get("character_rig") == "dhgen":
        from grail.pipelines.blender.dhgen import repose

        repose(parent_obj)
        return
    print("Applying relaxed pose to character...")

    # Find the armature among the children
    armature = None
    for child in parent_obj.children:
        if child.type == "ARMATURE":
            armature = child
            break
    if not armature:
        print("Warning: No armature found in RenderPeople character, cannot apply pose")
        return
    print(f"Found armature: {armature.name}")

    # Switch to pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Get the pose bones
    pose_bones = armature.pose.bones

    # List available bones for debugging
    # print(f"Available bones in armature:")
    # for bone in pose_bones:
    #     print(f"  - {bone.name}")

    # if rand_seed:
    #     random.seed(rand_seed)

    arm_bones = {
        "LeftShoulder": [0, -10, 0],
        "RightShoulder": [0, -10, 0],
        "LeftArm": [0, -65, 0],
        "RightArm": [0, -65, 0],
        "LeftForeArm": [45, -15, 0],
        "RightForeArm": [45, -15, 0],
    }
    for bone_name, angle_deg in arm_bones.items():
        bone = pose_bones[bone_name]
        bone.rotation_mode = "XYZ"
        if rand_seed and bone_name in ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm"]:
            angle_deg[2] += random.uniform(-10, 10)
        bone.rotation_euler = (
            math.radians(angle_deg[0]),
            math.radians(angle_deg[1]),
            math.radians(angle_deg[2]),
        )

    # Apply the pose by inserting keyframes (ensures the pose is saved)
    for bone in pose_bones:
        bone.keyframe_insert(data_path="rotation_euler")

    # Update the pose
    bpy.context.view_layer.update()

    # Return to object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Update the armature
    armature.update_tag()
    bpy.context.view_layer.update()

    print("Pose applied successfully")


def repose_renderpeople_from_saved(parent_obj, pose_file, ground_z=0.0):
    """
    Repose a RenderPeople character from a saved pose file (simple format).

    Loads bone names and quaternions directly - no axis-angle conversion or
    coordinate remapping. Use save_renderpeople_pose() to create these files.

    Supports both single-pose files (quaternions shape (N_bones, 4)) and
    multi-pose files from save_character_poses.py (shape (N_poses, N_bones, 4)).
    For multi-pose files, a random pose is selected.

    Args:
        parent_obj: The parent object containing the RenderPeople character
        pose_file: Path to a .npz file saved by save_renderpeople_pose() or
                   save_character_poses.py
        ground_z: Target Z coordinate for the character's feet after reposing
    """
    print(f"Reposing character from saved pose: {pose_file}")

    armature = None
    for child in parent_obj.children:
        if child.type == "ARMATURE":
            armature = child
            break
    if not armature:
        print("Warning: No armature found in RenderPeople character, cannot apply pose")
        return

    data = np.load(pose_file, allow_pickle=True)
    bone_names = data["bone_names"]
    quaternions = data["quaternions"]

    if quaternions.ndim == 3:
        idx = random.randint(0, quaternions.shape[0] - 1)
        print(f"  Multi-pose file with {quaternions.shape[0]} poses, randomly selected index {idx}")
        quaternions = quaternions[idx]

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    pose_bones = armature.pose.bones

    n_applied = 0
    for i, bone_name in enumerate(bone_names):
        name = str(bone_name)  # handle numpy string types
        if name not in pose_bones:
            continue
        bone = pose_bones[name]
        q = quaternions[i]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = mathutils.Quaternion((q[0], q[1], q[2], q[3]))
        n_applied += 1

    for bone in pose_bones:
        bone.keyframe_insert(data_path="rotation_quaternion")

    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.update_tag()
    bpy.context.view_layer.update()

    # Re-ground: find lowest vertex after pose deformation and shift parent
    # so the feet touch the ground plane (Z=0 relative to original position).
    min_z = float("inf")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for child in parent_obj.children:
        if child.type == "MESH":
            eval_obj = child.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            for vertex in mesh.vertices:
                world_vertex = eval_obj.matrix_world @ vertex.co
                min_z = min(min_z, world_vertex.z)
            eval_obj.to_mesh_clear()
    if min_z != float("inf"):
        parent_obj.location.z -= min_z - ground_z
        bpy.context.view_layer.update()
        print(
            f"Re-grounded character: shifted Z by {-(min_z - ground_z):.4f} (ground_z={ground_z:.4f})"
        )

    print(f"Pose applied successfully ({n_applied} bones)")


def load_renderpeople_from_name(
    character_name,
    position=(0, 0, 0),
    rotation=(0, 0, 0),
    scale=(1, 1, 1),
    rand_seed=None,
    character_dir="data/body_profiles/unitree_g1/characters",
    texture_dir="data/body_profiles/unitree_g1/characters",
    usd_file_path=None,
):
    """
    Convenience function to load a RenderPeople character by name

    Args:
        character_name (str): Name of the character (e.g., "aaron", "alexandra", "emma")
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians
        scale (tuple): (x, y, z) scale factors
        character_dir (str): Root directory containing RenderPeople USD folders
        texture_dir (str): Directory containing diffuse texture files
        usd_file_path (str, optional): Exact USD asset, bypassing name lookup

    Returns:
        object: The imported Blender object(s)
    """
    parent_obj = load_renderpeople_asset(
        character_name=character_name,
        usd_file_path=usd_file_path,
        position=position,
        rotation=rotation,
        scale=scale,
        character_dir=character_dir,
        texture_dir=texture_dir,
    )

    # Change character pose from T-pose to relaxed pose
    repose_renderpeople(parent_obj, rand_seed)

    return parent_obj


def load_g1_robot(
    xml_path,
    position=(0, 0, 0),
    rotation=(0, 0, 0),
    scale=(1, 1, 1),
):
    """
    Load a G1 robot model from MuJoCo XML and place it in the Blender scene at rest pose.

    Mimics the approach from src/visualization/vis_g1_data.py to create individual
    geom meshes with proper materials, parented under a single empty object.

    Args:
        xml_path (str): Path to MuJoCo XML G1 robot model
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians
        scale (tuple): (x, y, z) scale factors

    Returns:
        object: The parent Blender empty object containing all G1 robot mesh parts
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required to load G1 robot. Install with: pip install trimesh")

    try:
        import mujoco
    except ImportError:
        raise ImportError("mujoco is required to load G1 robot. Install with: pip install mujoco")

    raise NotImplementedError("G1 robot rendering requires vis_g1_data module (removed)")

    print(f"Loading G1 robot from: {xml_path}")

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"G1 XML file not found: {xml_path}")

    # Load the G1 model
    robot_model = G1Model(xml_path)  # noqa: F821

    # Use MuJoCo FK to get rest pose (qpos = 0)
    print("Computing rest pose via MuJoCo forward kinematics...")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Set qpos to zeros (rest pose)
    mj_data.qpos[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Get body names from G1Model (same order as used in mesh creation)
    g1_body_names = robot_model.get_body_names()

    # Create mapping from G1Model body index to MuJoCo body ID
    mj_body_name_to_id = {}
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        name = name if name else f"body_{i}"
        mj_body_name_to_id[name] = i

    # Extract positions and rotation matrices for rest pose
    num_bodies = len(g1_body_names)
    frame_positions = np.zeros((num_bodies, 3))
    frame_rotations = np.zeros((num_bodies, 3, 3))

    for g1_idx, g1_name in enumerate(g1_body_names):
        if g1_name in mj_body_name_to_id:
            mj_idx = mj_body_name_to_id[g1_name]
            frame_positions[g1_idx] = mj_data.xpos[mj_idx].copy()
            frame_rotations[g1_idx] = mj_data.xmat[mj_idx].reshape(3, 3).copy()
        else:
            frame_positions[g1_idx] = np.zeros(3)
            frame_rotations[g1_idx] = np.eye(3)

    print(f"  Rest pose computed for {num_bodies} bodies")

    # Create individual geom meshes with colors
    geom_meshes = []  # List of (trimesh_mesh, rgba, geom_name)

    for body_idx, body_name in enumerate(g1_body_names):
        if body_idx >= len(frame_positions):
            break

        body_pos = frame_positions[body_idx]
        body_rot = frame_rotations[body_idx]

        if body_name not in robot_model.bodies:
            continue

        body_info = robot_model.bodies[body_name]
        for geom_name in body_info["geoms"]:
            if geom_name in robot_model.geoms:
                geom_info = robot_model.geoms[geom_name]
                geom_mesh, geom_color = create_geom_mesh(  # noqa: F821
                    geom_info,
                    body_pos,
                    body_rot,
                    robot_model.mesh_dir,
                    robot_model.mesh_assets,
                )
                if geom_mesh is not None:
                    geom_meshes.append((geom_mesh, geom_color, geom_name))

    if not geom_meshes:
        raise RuntimeError("Failed to create any G1 robot meshes")

    print(f"  Created {len(geom_meshes)} geom meshes")

    # Store objects before adding G1 parts to identify them later
    objects_before = set(bpy.data.objects)

    # Create parent empty object
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    parent_obj = bpy.context.active_object
    parent_obj.name = "G1_Robot"

    # Create Blender mesh objects for each geom
    for idx, (geom_mesh, geom_color, geom_name) in enumerate(geom_meshes):
        vertices = np.array(geom_mesh.vertices)
        faces = np.array(geom_mesh.faces)

        # Create Blender mesh
        mesh_data = bpy.data.meshes.new(f"G1_{geom_name}")
        obj_name = f"G1_{geom_name}"
        mesh_obj = bpy.data.objects.new(obj_name, mesh_data)

        faces_list = [tuple(face) for face in faces]
        mesh_data.from_pydata(vertices.tolist(), [], faces_list)
        mesh_data.update()

        # Link to scene
        bpy.context.scene.collection.objects.link(mesh_obj)

        # Apply material with geom-specific color
        mat = bpy.data.materials.new(name=f"G1_Material_{geom_name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled:
            principled.inputs["Base Color"].default_value = tuple(geom_color)
            principled.inputs["Metallic"].default_value = 0.5
            principled.inputs["Roughness"].default_value = 0.4

        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = mat
        else:
            mesh_obj.data.materials.append(mat)

        # Parent to the empty
        mesh_obj.parent = parent_obj
        mesh_obj.parent_type = "OBJECT"
        mesh_obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

    # Apply transformations to the parent
    parent_obj.rotation_euler = rotation
    parent_obj.scale = scale
    parent_obj.location = position

    # Update scene
    bpy.context.view_layer.update()

    print(f"G1 robot loaded with {len(geom_meshes)} mesh parts")
    print(f"  Position: {position}")
    print(f"  Rotation: {rotation}")
    print(f"  Scale: {scale}")

    return parent_obj


def check_textures_loaded(obj):
    """
    Check if textures are properly loaded for the imported object

    Args:
        obj: The imported Blender object

    Returns:
        bool: True if textures are found, False otherwise
    """
    print("=" * 50)
    print("TEXTURE LOADING CHECK")
    print("=" * 50)

    if not obj or not hasattr(obj, "data") or not hasattr(obj.data, "materials"):
        print("No materials found on object")
        return False

    materials = obj.data.materials
    print(f"Found {len(materials)} material(s) on object:")

    textures_found = False

    for i, material in enumerate(materials):
        if material is None:
            print(f"  Material {i}: None")
            continue

        print(f"  Material {i}: {material.name}")

        # Check for Shader Editor nodes (Blender 2.8+)
        if hasattr(material, "node_tree") and material.node_tree:
            print("    Using node-based materials")
            nodes = material.node_tree.nodes

            for node in nodes:
                if node.type == "TEX_IMAGE":
                    print(f"    Image Texture Node: {node.name}")
                    if node.image:
                        print(f"      Image: {node.image.name}")
                        print(f"      File path: {node.image.filepath}")
                        print(f"      Size: {node.image.size[0]}x{node.image.size[1]}")
                        textures_found = True
                    else:
                        print("      No image loaded in texture node")

        # Check for legacy texture slots (Blender 2.7x style)
        if hasattr(material, "texture_slots"):
            texture_slots = [slot for slot in material.texture_slots if slot is not None]
            if texture_slots:
                print(f"    Found {len(texture_slots)} texture slot(s):")
                for j, slot in enumerate(texture_slots):
                    if slot and slot.texture:
                        print(f"      Slot {j}: {slot.texture.name} (Type: {slot.texture.type})")
                        if slot.texture.type == "IMAGE" and slot.texture.image:
                            print(f"        Image: {slot.texture.image.name}")
                            print(f"        File path: {slot.texture.image.filepath}")
                            textures_found = True
                        else:
                            print("        No image in texture")
            else:
                print("    No texture slots found")

        # Check material properties
        if hasattr(material, "diffuse_color"):
            print(f"    Diffuse color: {material.diffuse_color[:3]}")

    print("=" * 50)
    if textures_found:
        print("✅ TEXTURES FOUND - Materials have image textures loaded")
    else:
        print("❌ NO TEXTURES FOUND - Materials may be using solid colors only")
    print("=" * 50)

    return textures_found


def fix_texture_paths(obj, obj_file_path):
    """
    Fix relative texture paths to absolute paths after OBJ import

    Args:
        obj: The imported Blender object
        obj_file_path: Path to the original OBJ file
    """
    print("=" * 50)
    print("FIXING TEXTURE PATHS")
    print("=" * 50)

    obj_dir = os.path.dirname(os.path.abspath(obj_file_path))
    print(f"OBJ directory: {obj_dir}")

    fixed_count = 0

    # Fix all images in the scene
    for image in bpy.data.images:
        if image.filepath and image.filepath.startswith("//"):
            print(f"  Fixing: {image.name}")
            print(f"    Old Blender path: {image.filepath}")

            # Blender converts relative paths to //../../../../../../path/to/file
            # We need to extract the meaningful part (after all the ../)
            relative_path = image.filepath[2:]  # Remove "//" prefix

            # Split the path and remove all ".." parts to get the meaningful path
            path_parts = relative_path.replace("\\", "/").split("/")

            # Find where the actual path starts (skip all ".." entries)
            meaningful_parts = []
            skip_dots = True
            for part in path_parts:
                if skip_dots and part == "..":
                    continue
                else:
                    skip_dots = False
                    meaningful_parts.append(part)

            if not meaningful_parts:
                print(f"    ❌ Could not extract meaningful path from: {relative_path}")
                continue

            # The meaningful path should be relative to the OBJ directory
            # For example: "data/ComAsset/ladder/.../images/texture.png"
            # We need to find where this intersects with our obj_dir

            # Try different strategies to locate the file:
            found = False
            absolute_path = None

            # Strategy 1: Find common base path
            # If the meaningful path starts with "data/", find "data/" in obj_dir
            for i in range(len(meaningful_parts)):
                partial_path = os.path.join(*meaningful_parts[i:])
                test_path = os.path.join(obj_dir, partial_path)
                if os.path.exists(test_path):
                    absolute_path = test_path
                    found = True
                    break

            # Strategy 2: Just use the last few parts relative to obj_dir
            # (e.g., "images/texture.png" relative to the OBJ file)
            if not found:
                for num_parts in [2, 1]:  # Try "images/texture.png", then just "texture.png"
                    if len(meaningful_parts) >= num_parts:
                        partial_path = os.path.join(*meaningful_parts[-num_parts:])
                        test_path = os.path.join(obj_dir, partial_path)
                        if os.path.exists(test_path):
                            absolute_path = test_path
                            found = True
                            break

            # Strategy 3: Search common texture subdirectories
            if not found:
                texture_filename = meaningful_parts[-1]  # Just the filename
                for subdir in ["images", "textures", "texture", ""]:
                    test_path = os.path.join(obj_dir, subdir, texture_filename)
                    if os.path.exists(test_path):
                        absolute_path = test_path
                        found = True
                        break

            if found and absolute_path:
                print(f"    New path: {absolute_path}")
                image.filepath = absolute_path
                image.reload()  # Reload the image data
                print("    ✅ Fixed and reloaded successfully")
                fixed_count += 1
            else:
                # Show what we tried
                print("    ❌ File not found. Tried:")
                print(f"       - Relative to OBJ dir: {meaningful_parts}")
                print(f"       - In OBJ directory: {obj_dir}")

    print(f"Fixed {fixed_count} texture path(s)")
    print("=" * 50)

    return fixed_count > 0


def load_obj_asset(obj_file_path, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    """
    Load an OBJ file and place it in the scene with specified transform

    Args:
        obj_file_path (str): Path to the .obj file
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians
        scale (tuple): (x, y, z) scale factors

    Returns:
        object: The imported Blender object
    """
    if not os.path.exists(obj_file_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_file_path}")

    print(f"Loading OBJ asset: {obj_file_path}")

    # Clear selection
    bpy.ops.object.select_all(action="DESELECT")

    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=obj_file_path)

    # Get the imported object (should be the last selected object)
    imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

    if imported_obj:
        # Fix texture paths first
        fix_texture_paths(imported_obj, obj_file_path)

        # Check if textures are loaded (after fixing paths)
        # check_textures_loaded(imported_obj)

        # First set rotation and scale
        imported_obj.rotation_euler = rotation
        print(f"Object rotated to: {rotation} radians")

        imported_obj.scale = scale
        print(f"Object scaled to: {scale}")

        # Update the scene to apply transforms
        bpy.context.view_layer.update()

        # Calculate minimum Z coordinate of vertices after scaling and rotation
        mesh = imported_obj.data
        min_z = float("inf")

        for vertex in mesh.vertices:
            # Transform vertex to world coordinates
            world_vertex = imported_obj.matrix_world @ vertex.co
            min_z = min(min_z, world_vertex.z)

        print(f"Object minimum Z coordinate: {min_z:.3f}")

        # Adjust position to place bottom of object at desired Z
        adjusted_position = (position[0], position[1], position[2] - min_z)
        imported_obj.location = adjusted_position
        print(f"Object positioned at: {adjusted_position} (adjusted for bottom placement)")

        # Final update
        bpy.context.view_layer.update()

        print(f"OBJ asset '{imported_obj.name}' loaded and positioned successfully")
        return imported_obj
    else:
        raise RuntimeError("Failed to import OBJ file or no object was created")


def load_static_objects(static_objects_config):
    """
    Load static objects from scene config, mimicking the loaded_obj pattern.

    Args:
        static_objects_config (dict): Dictionary of static objects with structure:
            {
                "object_key": {
                    "name": "filename.obj",
                    "pos": [x, y, z],
                    "rot": [rx, ry, rz],  # in degrees
                    "scale": [sx, sy, sz],
                    "opacity": 0.5,  # optional, 0.0-1.0 (default: 1.0 fully opaque)
                },
                ...
            }

    Returns:
        dict: Dictionary mapping object keys to loaded Blender objects
    """
    loaded_static_objects = {}

    for obj_key, obj_config in static_objects_config.items():
        obj_name = obj_config["name"]
        obj_path = os.path.join("data/Scene", obj_name)

        if not os.path.exists(obj_path):
            print(f"Warning: Static object file not found: {obj_path}, skipping {obj_key}")
            continue

        position = obj_config.get("pos", [0, 0, 0])
        rotation_deg = obj_config.get("rot", [0, 0, 0])
        scale = obj_config.get("scale", [1, 1, 1])

        # Convert rotation from degrees to radians
        rotation_radians = [math.radians(deg) for deg in rotation_deg]

        print(f"Loading static object '{obj_key}': {obj_name}")
        print(f"  Position: {position}")
        print(f"  Rotation: {rotation_deg} degrees")
        print(f"  Scale: {scale}")

        # Load the object based on file extension
        if obj_name.endswith(".obj"):
            loaded_obj = load_obj_asset(
                obj_path,
                position=position,
                rotation=rotation_radians,
                scale=scale,
            )
        elif obj_name.endswith(".glb"):
            loaded_obj = load_glb_asset(
                obj_path,
                position=position,
                rotation=rotation_radians,
                scale=scale,
            )
        else:
            print(f"Warning: Unsupported file format for {obj_name}, skipping {obj_key}")
            continue

        # Apply opacity if specified
        opacity = obj_config.get("opacity", None)
        if opacity is not None:
            set_object_opacity(loaded_obj, opacity)
            print(f"  Opacity: {opacity}")

        # add object path to loaded_obj
        loaded_obj["path"] = obj_path

        loaded_obj["obj_on_table"] = obj_config.get("obj_on_table", True)

        loaded_static_objects[obj_key] = loaded_obj
        print(f"Static object '{obj_key}' loaded successfully")

    return loaded_static_objects


def set_object_opacity(obj, opacity):
    """
    Set the opacity/transparency of an object's materials.

    Args:
        obj: Blender object to modify
        opacity (float): Opacity value between 0.0 (fully transparent) and 1.0 (fully opaque)
    """
    if obj is None:
        print("Warning: Cannot set opacity on None object")
        return

    # Clamp opacity to valid range
    opacity = max(0.0, min(1.0, opacity))

    # Handle mesh objects
    if obj.type == "MESH":
        mesh = obj.data

        # If no materials exist, create one
        if len(mesh.materials) == 0:
            mat = bpy.data.materials.new(name=f"{obj.name}_Material")
            mesh.materials.append(mat)

        # Apply opacity to all materials
        for mat in mesh.materials:
            if mat is None:
                continue

            # Enable nodes if not already
            mat.use_nodes = True
            mat.blend_method = "BLEND"  # Enable alpha blending
            mat.shadow_method = "HASHED"  # Better shadow handling for transparent objects

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Find or create Principled BSDF
            principled = None
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled = node
                    break

            if principled is None:
                # Create Principled BSDF if not found
                principled = nodes.new(type="ShaderNodeBsdfPrincipled")
                principled.location = (0, 0)

                # Find or create output node
                output = None
                for node in nodes:
                    if node.type == "OUTPUT_MATERIAL":
                        output = node
                        break

                if output is None:
                    output = nodes.new(type="ShaderNodeOutputMaterial")
                    output.location = (300, 0)

                # Connect principled to output
                links.new(principled.outputs["BSDF"], output.inputs["Surface"])

            # Remove any existing links to the Alpha input so default_value takes effect
            # (OBJ importer may connect texture Alpha to this input, overriding default_value)
            if "Alpha" in principled.inputs:
                for link in list(links):
                    if link.to_socket == principled.inputs["Alpha"]:
                        links.remove(link)
                        print("    Removed existing link to Alpha input")
                principled.inputs["Alpha"].default_value = opacity

            print(f"    Set opacity {opacity} on material: {mat.name}")

    # Handle empty objects with children (e.g., from GLB imports)
    elif obj.type == "EMPTY":
        for child in obj.children:
            set_object_opacity(child, opacity)


def load_glb_asset(glb_file_path, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    """
    Load a GLB file and place it in the scene with specified transform
    GLB files may contain vertex colors which will be automatically set up for rendering
    Automatically converts from X-up to Z-up coordinate system

    Args:
        glb_file_path (str): Path to the .glb file
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians (applied after X-up to Z-up conversion)
        scale (tuple): (x, y, z) scale factors

    Returns:
        object: The imported Blender object
    """
    if not os.path.exists(glb_file_path):
        raise FileNotFoundError(f"GLB file not found: {glb_file_path}")

    print(f"Loading GLB asset: {glb_file_path}")

    # Clear selection
    bpy.ops.object.select_all(action="DESELECT")

    # Store objects before import to identify new ones
    objects_before = set(bpy.data.objects)

    # Import the GLB file
    bpy.ops.import_scene.gltf(filepath=glb_file_path)

    # Get the newly imported objects
    objects_after = set(bpy.data.objects)
    imported_objects = list(objects_after - objects_before)

    if not imported_objects:
        raise RuntimeError("Failed to import GLB file or no objects were created")

    # Find the main mesh object
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]

    if not mesh_objects:
        raise RuntimeError("No mesh objects found in GLB file")

    # Use the first mesh object as the main object
    imported_obj = mesh_objects[0]

    # If there are multiple objects, create a parent for them
    if len(imported_objects) > 1:
        # Create an empty parent
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        parent_obj = bpy.context.active_object
        parent_obj.name = f"GLB_{os.path.basename(glb_file_path).split('.')[0]}"

        # Parent all imported objects to the empty
        for obj in imported_objects:
            obj.parent = parent_obj
            obj.parent_type = "OBJECT"
            obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

        # Use the parent for transformations
        transform_obj = parent_obj
    else:
        transform_obj = imported_obj

    print(f"Imported {len(imported_objects)} object(s) from GLB file")

    # Set up materials to use vertex colors for all mesh objects
    for mesh_obj in mesh_objects:
        if mesh_obj.type == "MESH":
            mesh = mesh_obj.data

            # Check if the mesh has color attributes (vertex colors)
            if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
                print(f"  Setting up vertex colors for mesh: {mesh_obj.name}")
                print(f"  Found {len(mesh.color_attributes)} color attribute(s)")

                # Get or create material
                if len(mesh_obj.data.materials) == 0:
                    # Create new material if none exists
                    mat = bpy.data.materials.new(name=f"{mesh_obj.name}_VertexColor")
                    mesh_obj.data.materials.append(mat)
                else:
                    # Use existing material
                    mat = mesh_obj.data.materials[0]

                # Enable nodes for the material
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                # Clear existing nodes
                nodes.clear()

                # Create shader nodes
                output_node = nodes.new(type="ShaderNodeOutputMaterial")
                output_node.location = (400, 0)

                bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
                bsdf_node.location = (0, 0)

                # Create Color Attribute node to read vertex colors
                color_attr_node = nodes.new(type="ShaderNodeVertexColor")
                color_attr_node.location = (-400, 0)

                # Use the first color attribute by default
                color_attr_node.layer_name = mesh.color_attributes[0].name
                print(f"  Using color attribute: {mesh.color_attributes[0].name}")

                # Connect nodes: Color Attribute -> BSDF Base Color -> Output
                links.new(color_attr_node.outputs["Color"], bsdf_node.inputs["Base Color"])
                links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

                print("  ✓ Vertex color material set up successfully")
            else:
                print(f"  No vertex colors found for mesh: {mesh_obj.name}")

    # Convert from X-up to Z-up coordinate system
    # Rotate -90 degrees around Y-axis to convert X-up to Z-up
    x_to_z_rotation = (0, math.pi / 2, 0)  # -90 degrees around Y-axis

    # Apply user-specified rotation on top of the coordinate system conversion
    # Combine rotations by converting to matrix and back
    base_rotation = mathutils.Euler(x_to_z_rotation, "XYZ")
    user_rotation = mathutils.Euler(rotation, "XYZ")
    combined_rotation = (base_rotation.to_matrix() @ user_rotation.to_matrix()).to_euler()

    # transform_obj.rotation_euler = combined_rotation
    transform_obj.rotation_euler = rotation
    print(f"Object rotated to: {combined_rotation} radians (includes X-up to Z-up conversion)")

    transform_obj.scale = scale
    print(f"Object scaled to: {scale}")

    # Update the scene to apply transforms
    bpy.context.view_layer.update()

    # Calculate minimum Z coordinate of vertices after scaling and rotation
    min_z = float("inf")

    # Check all mesh objects for minimum Z
    for mesh_obj in mesh_objects:
        if mesh_obj.type == "MESH":
            mesh = mesh_obj.data
            for vertex in mesh.vertices:
                # Transform vertex to world coordinates
                world_vertex = mesh_obj.matrix_world @ vertex.co
                min_z = min(min_z, world_vertex.z)

    print(f"Object minimum Z coordinate: {min_z:.3f}")

    # Adjust position to place bottom of object at desired Z
    adjusted_position = (position[0], position[1], position[2] - min_z)
    transform_obj.location = adjusted_position
    print(f"Object positioned at: {adjusted_position} (adjusted for bottom placement)")

    # Final update
    bpy.context.view_layer.update()

    print(f"GLB asset '{transform_obj.name}' loaded and positioned successfully")
    return transform_obj


def resolve_renderpeople_usd(
    character_name=None,
    usd_file_path=None,
    character_dir="data/body_profiles/unitree_g1/characters",
):
    """Resolve an exact asset or a legacy name without importing it into Blender.

    The public default is the bundled G1 DHGen character. Historical name lookup
    remains available when a caller explicitly selects a legacy asset directory.
    """
    if usd_file_path is not None:
        return os.fspath(usd_file_path)
    if character_name is None:
        raise ValueError("Either character_name or usd_file_path must be provided")

    repo = Path(__file__).resolve().parents[3]
    bundled = repo / "data/body_profiles/unitree_g1/characters"
    if (
        character_name.lower()
        in {
            "unitree_g1",
            "jason_g1_rigged_001",
            "unitree_g1_dhgen_00008",
            "dhgen_00008",
        }
        and Path(character_dir).resolve() == bundled.resolve()
    ):
        return str(bundled / "dhgen_00008" / "c416aab4-cf3f-11f1-bd53-1929f989c6a9.usd")

    matching_dirs = [
        name for name in os.listdir(character_dir) if character_name.lower() in name.lower()
    ]
    if not matching_dirs:
        raise FileNotFoundError(f"No character found matching '{character_name}'")
    selected_dir = matching_dirs[0]
    if len(matching_dirs) > 1:
        print(f"Multiple matches found for '{character_name}', using: {selected_dir}")
    usd_files = sorted(Path(character_dir, selected_dir).glob("*.usd"))
    if len(usd_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one USD under {Path(character_dir, selected_dir)}, "
            f"found {len(usd_files)}"
        )
    return str(usd_files[0])


def load_renderpeople_asset(
    character_name=None,
    usd_file_path=None,
    position=(0, 0, 0),
    rotation=(0, 0, 0),
    scale=(1, 1, 1),
    character_dir="data/body_profiles/unitree_g1/characters",
    texture_dir="data/body_profiles/unitree_g1/characters",
):
    """
    Load a RenderPeople USD file and place it in the scene with specified transform

    Args:
        character_name (str, optional): RenderPeople name, or unitree_g1 /
                                       jason_g1_rigged_001 for the packaged G1 human
        usd_file_path (str, optional): Direct path to the .usd file
                                      Either character_name or usd_file_path must be provided
        position (tuple): (x, y, z) position in world coordinates
        rotation (tuple): (x, y, z) rotation in radians
        scale (tuple): (x, y, z) scale factors
        character_dir (str): Root directory containing RenderPeople USD folders
        texture_dir (str): Directory containing diffuse texture files

    Returns:
        object: The imported Blender object(s)
    """

    usd_file_path = resolve_renderpeople_usd(character_name, usd_file_path, character_dir)

    if not os.path.exists(usd_file_path):
        raise FileNotFoundError(f"USD file not found: {usd_file_path}")

    print(f"Loading character USD asset: {usd_file_path}")
    from grail.pipelines.blender.dhgen import open_dhgen_stage, prepare_import

    dhgen_stage = open_dhgen_stage(usd_file_path)

    # Clear selection
    bpy.ops.object.select_all(action="DESELECT")

    # Store objects before import to identify new ones
    objects_before = set(bpy.data.objects)

    # Import the USD file
    # Note: Blender 3.0+ has built-in USD support
    try:
        bpy.ops.wm.usd_import(filepath=usd_file_path)
    except AttributeError:
        # Fallback for older Blender versions or if USD addon is not enabled
        print("USD import operator not found. Trying to enable USD addon...")
        bpy.ops.preferences.addon_enable(module="io_scene_usd")
        bpy.ops.wm.usd_import(filepath=usd_file_path)

    # Get the newly imported objects
    objects_after = set(bpy.data.objects)
    imported_objects = list(objects_after - objects_before)

    if not imported_objects:
        raise RuntimeError("Failed to import USD file or no objects were created")

    print(f"Imported {len(imported_objects)} object(s) from USD file")

    # Debug: List all imported objects
    print("Imported objects:")
    for obj in imported_objects:
        print(f"  - {obj.name} (Type: {obj.type}, Location: {obj.location})")

    # Find the main mesh object (usually the body)
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]

    # Create an empty parent for all imported objects
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    parent_obj = bpy.context.active_object
    parent_obj.name = f"Character_{os.path.basename(usd_file_path).split('.')[0]}"

    if dhgen_stage is not None:
        prepare_import(dhgen_stage, imported_objects, parent_obj, usd_file_path)
        parent_obj.rotation_euler = rotation
        parent_obj.scale = scale
        parent_obj.location = position
        bpy.context.view_layer.update()
        return parent_obj

    # Parent all imported objects to the empty
    for obj in imported_objects:
        # Keep their relative positions
        obj.parent = parent_obj
        obj.parent_type = "OBJECT"
        obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

    # Apply transforms to the parent
    transform_obj = parent_obj

    # Apply transformations
    transform_obj.rotation_euler = rotation
    print(f"Character rotated to: {rotation} radians")

    transform_obj.scale = scale
    print(f"Character scaled to: {scale}")

    # Update the scene to apply transforms
    bpy.context.view_layer.update()

    # For now, just place at the specified position
    # The USD models should already be positioned correctly relative to origin
    transform_obj.location = position
    print(f"Character positioned at: {position}")

    # Final update
    bpy.context.view_layer.update()

    # Determine the character base name for texture loading
    character_base_name = None
    if usd_file_path:
        # Extract base name from USD file path
        usd_filename = os.path.basename(usd_file_path)
        # Remove .usd extension and _yup_t suffix if present
        character_base_name = usd_filename.replace(".usd", "").replace("_yup_t", "")

    # Ensure all imported objects are visible
    for obj in imported_objects:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_set(False)

        # Try to fix materials for mesh objects
        if obj.type == "MESH" and obj.data.materials:
            print(f"Checking materials for {obj.name}...")
            for mat_slot in obj.material_slots:
                if mat_slot.material:
                    mat = mat_slot.material
                    print(f"  Material: {mat.name}")

                    try:
                        # Set up a basic material if the texture is missing
                        if mat.use_nodes:
                            # Try to load texture from texture_dir if exists
                            texture_loaded = False
                            if character_base_name:
                                # Look for diffuse texture file
                                # HACK: reuse the same texture for g1 proportioned character model
                                texture_filename = f"{character_base_name}_dif.png".replace(
                                    "_g1", ""
                                )
                                texture_path = os.path.join(texture_dir, texture_filename)

                                if os.path.exists(texture_path):
                                    print(f"    Found texture file: {texture_path}")

                                    # Find or create texture node
                                    tex_node = None
                                    for node in mat.node_tree.nodes:
                                        if node.type == "TEX_IMAGE":
                                            tex_node = node
                                            break

                                    if not tex_node:
                                        tex_node = mat.node_tree.nodes.new(
                                            type="ShaderNodeTexImage"
                                        )
                                        tex_node.location = (-300, 0)

                                    # Load the texture
                                    tex_node.image = bpy.data.images.load(texture_path)
                                    print(f"    Loaded texture: {texture_filename}")

                                    # Connect to Principled BSDF
                                    principled = None
                                    for node in mat.node_tree.nodes:
                                        if node.type == "BSDF_PRINCIPLED":
                                            principled = node
                                            break

                                    if principled and not principled.inputs["Base Color"].is_linked:
                                        mat.node_tree.links.new(
                                            tex_node.outputs["Color"],
                                            principled.inputs["Base Color"],
                                        )
                                        print("    Connected texture to material")

                                    texture_loaded = True
                                else:
                                    print(f"    Texture file not found: {texture_path}")

                            # Check if there are any texture nodes with missing images (if texture wasn't loaded)
                            if not texture_loaded:
                                nodes_to_remove = []
                                for node in mat.node_tree.nodes:
                                    if node.type == "TEX_IMAGE":
                                        if node.image is None:
                                            print(
                                                "    Missing texture image node (no image assigned)"
                                            )
                                            nodes_to_remove.append(node)
                                        elif not os.path.exists(
                                            bpy.path.abspath(node.image.filepath)
                                        ):
                                            print(
                                                f"    Missing texture file: {node.image.filepath}"
                                            )
                                            nodes_to_remove.append(node)

                                # Remove broken texture nodes
                                for node in nodes_to_remove:
                                    mat.node_tree.nodes.remove(node)
                            # Ensure there's a proper material setup
                            nodes = mat.node_tree.nodes
                            links = mat.node_tree.links

                            # Find or create Principled BSDF
                            principled = None
                            for node in nodes:
                                if node.type == "BSDF_PRINCIPLED":
                                    principled = node
                                    break

                            if not principled:
                                principled = nodes.new(type="ShaderNodeBsdfPrincipled")
                                principled.location = (0, 0)

                            # Set a skin-like color if no texture is connected
                            if not principled.inputs["Base Color"].is_linked:
                                # Set a neutral skin tone color
                                principled.inputs["Base Color"].default_value = (
                                    0.8,
                                    0.65,
                                    0.5,
                                    1.0,
                                )
                                principled.inputs["Roughness"].default_value = 0.5

                                # Handle different Blender versions - Specular was renamed in 3.0+
                                if "Specular" in principled.inputs:
                                    principled.inputs["Specular"].default_value = 0.3
                                elif "Specular IOR Level" in principled.inputs:
                                    principled.inputs["Specular IOR Level"].default_value = 0.3
                                elif "IOR" in principled.inputs:
                                    # Set IOR for skin-like material (1.4 is typical for skin)
                                    principled.inputs["IOR"].default_value = 1.4

                            # Find or create output node
                            output = None
                            for node in nodes:
                                if node.type == "OUTPUT_MATERIAL":
                                    output = node
                                    break

                            if not output:
                                output = nodes.new(type="ShaderNodeOutputMaterial")
                                output.location = (300, 0)

                            # Connect principled to output if not connected
                            if not output.inputs["Surface"].is_linked:
                                links.new(principled.outputs["BSDF"], output.inputs["Surface"])

                            print("    Applied default skin-tone material")
                    except Exception as e:
                        print(f"    Warning: Could not fix material: {e}")

    # Also ensure parent is visible
    transform_obj.hide_viewport = False
    transform_obj.hide_render = False
    transform_obj.hide_set(False)

    print("Character asset loaded and positioned successfully")
    return transform_obj


def create_white_texture_image(output_path, size=(512, 512)):
    """
    Create a white texture image

    Args:
        output_path (str): Path to save the white texture image
        size (tuple): Image size (width, height)
    """
    try:
        import cv2

        # Create white image
        white_image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        cv2.imwrite(output_path, white_image)
        print(f"Created white texture: {output_path}")
        return True
    except ImportError:
        try:
            from PIL import Image

            # Create white image using PIL
            white_image = Image.new("RGB", size, (255, 255, 255))
            white_image.save(output_path)
            print(f"Created white texture: {output_path}")
            return True
        except ImportError:
            print("Warning: Cannot create white texture - neither OpenCV nor PIL available")
            return False


def process_mtl_file_for_textures(mtl_file_path, mesh_output_dir):
    """
    Process MTL file to check for missing textures and create white textures if needed

    Args:
        mtl_file_path (str): Path to the MTL file
        mesh_output_dir (str): Directory where mesh files are copied
    """
    if not os.path.exists(mtl_file_path):
        return

    print(f"Processing MTL file: {mtl_file_path}")

    # Read MTL file and look for texture references
    texture_references = []
    mtl_content_lines = []

    with open(mtl_file_path, "r") as f:
        lines = f.readlines()

    # Parse MTL file to find texture references
    for line in lines:
        line = line.strip()
        mtl_content_lines.append(line)

        # Look for texture map directives
        if (
            line.startswith("map_Kd ")
            or line.startswith("map_Ka ")
            or line.startswith("map_Ks ")
            or line.startswith("map_Ke ")
            or line.startswith("map_Ns ")
            or line.startswith("map_d ")
            or line.startswith("map_bump ")
            or line.startswith("bump ")
        ):

            # Extract texture filename
            texture_name = line.split(" ", 1)[1].strip().strip("\"'")
            texture_references.append(texture_name)

    # Check if referenced textures exist
    missing_textures = []
    for texture_name in texture_references:
        # Check in source directory first
        source_texture_path = os.path.join(os.path.dirname(mtl_file_path), texture_name)
        dest_texture_path = os.path.join(mesh_output_dir, texture_name)

        if not os.path.exists(source_texture_path) and not os.path.exists(dest_texture_path):
            missing_textures.append(texture_name)
    # Create white textures for missing ones
    for texture_name in missing_textures:
        dest_texture_path = os.path.join(mesh_output_dir, texture_name)

        # Create directory if texture is in a subdirectory
        texture_dir = os.path.dirname(dest_texture_path)
        if texture_dir:
            os.makedirs(texture_dir, exist_ok=True)

        # Create white texture
        create_white_texture_image(dest_texture_path)

    # If MTL file has no texture references at all, create a default texture
    if not texture_references:
        print("MTL file has no texture references - adding default white texture")

        # Create default white texture
        default_texture_name = "texture.png"
        default_texture_path = os.path.join(mesh_output_dir, default_texture_name)

        if create_white_texture_image(default_texture_path):
            # Modify MTL file to reference the white texture
            dest_mtl_path = os.path.join(mesh_output_dir, os.path.basename(mtl_file_path))

            # Add map_Kd reference to the MTL file
            with open(dest_mtl_path, "a") as f:
                f.write(f"\nmap_Kd {default_texture_name}\n")

            print(f"Added default texture reference to MTL file: {default_texture_name}")


def copy_mesh_files(obj_file_path, mesh_output_dir):
    """
    Copy all files from the source mesh directory to the target directory
    and ensure MTL files have proper texture references

    Args:
        obj_file_path (str): Path to the source OBJ file
        mesh_output_dir (str): Directory to copy mesh files to
    """
    # Get the source directory containing the mesh files
    source_mesh_dir = os.path.dirname(obj_file_path)

    print(f"Copying all mesh files from: {source_mesh_dir}")
    print(f"To mesh directory: {mesh_output_dir}")

    # Create mesh output directory
    os.makedirs(mesh_output_dir, exist_ok=True)

    try:
        # Copy entire directory contents using shutil.copytree with dirs_exist_ok=True
        # This will copy all files and subdirectories while preserving structure
        for item in os.listdir(source_mesh_dir):
            source_item = os.path.join(source_mesh_dir, item)
            dest_item = os.path.join(mesh_output_dir, item)

            if os.path.isdir(source_item):
                # Copy directory recursively
                shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                print(f"Copied directory: {item}")
            else:
                # Copy file
                shutil.copy2(source_item, dest_item)
                print(f"Copied file: {item}")

        # Process all MTL files to ensure they have textures
        mtl_files = glob.glob(os.path.join(mesh_output_dir, "*.mtl"))

        if mtl_files:
            print(f"Found {len(mtl_files)} MTL file(s): {[os.path.basename(f) for f in mtl_files]}")
            for mtl_file in mtl_files:
                process_mtl_file_for_textures(mtl_file, mesh_output_dir)
        else:
            print(f"No MTL files found in: {mesh_output_dir}")

        print(f"Successfully copied all mesh files to: {mesh_output_dir}")

    except Exception as e:
        print(f"Error copying mesh files: {str(e)}")
        raise
