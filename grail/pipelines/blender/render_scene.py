# import blenderproc as bproc
#!/usr/bin/env python3
"""
Blender GPU Rendering Script

This script loads a .blend file and renders the scene using GPU acceleration.
It configures the render engine for optimal GPU performance and handles
common rendering scenarios.

Usage:
    python render_blender_scene.py --input scene.blend --output output.png

Requirements:
    - Blender installed as a Python module or accessible via command line
    - CUDA-compatible GPU for NVIDIA cards or OpenCL for AMD/other GPUs
"""

import argparse
import glob
import hashlib
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import bpy  # isort: skip  # must be imported before bmesh
import bmesh
import mathutils
import numpy as np
from mathutils.bvhtree import BVHTree

from grail.constants.image import FOCAL_LENGTH, HEIGHT, WIDTH
from grail.core.config import load_object_config, load_pipeline_config, load_scene_config
from grail.core.dataset import category2object, scene2blender
from grail.core.io import save_camera_intrinsics, save_init_rendering_data

# Import all functions from split modules
from grail.pipelines.blender.assets import *  # noqa: F401,F403
from grail.pipelines.blender.utils import *  # noqa: F401,F403


def main():
    """Main function to handle command line arguments and orchestrate rendering"""
    parser = argparse.ArgumentParser(description="Render Blender scene using GPU")

    # Input/Output arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="ComAsset",
        help="Dataset name. Meshes are looked up under data/<dataset>/<category>/.",
    )
    parser.add_argument("--category", type=str, default="barbell", help="Object category")
    parser.add_argument(
        "--character_name", type=str, default=None, help="Character name (use 'G1' for G1 robot)"
    )
    parser.add_argument(
        "--character_usd",
        type=str,
        default=None,
        help="Exact USD character asset; bypass name matching",
    )
    parser.add_argument(
        "--character_scale",
        type=float,
        nargs=3,
        default=None,
        help="Override external character scale (meter USD: 1 1 1)",
    )
    parser.add_argument(
        "--character_dir",
        type=str,
        default="data/body_profiles/unitree_g1/characters",
        help="Root directory containing character USD folders",
    )
    parser.add_argument(
        "--texture_dir",
        type=str,
        default="data/body_profiles/unitree_g1/characters",
        help="Directory containing character textures",
    )
    parser.add_argument(
        "--g1_xml_path",
        type=str,
        default="data/unitree_g1/g1_mocap_29dof_with_hands.xml",
        help="Path to G1 MuJoCo XML model (used when character_name=G1)",
    )
    parser.add_argument("--scene", type=str, default=None, help="Scene name")
    parser.add_argument(
        "--rand_scene_seed",
        type=int,
        default=None,
        help="seed to randomize character start position, start pose, and camera position",
    )
    parser.add_argument(
        "--no_rand_seed",
        action="store_true",
        default=False,
        help="Skip random.seed() calls so randomization is non-deterministic",
    )
    parser.add_argument("--base_seed", type=int, default=42, help="base seed for randomization")

    # Save directories
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="base directory to save results",
    )
    parser.add_argument(
        "--image_save_dir",
        type=str,
        default="generation/asset_renders",
        help="Directory to save rendered images",
    )
    parser.add_argument(
        "--camera_dir",
        type=str,
        default="generation/cameras",
        help="Directory to save camera data",
    )
    parser.add_argument(
        "--depth_save_dir",
        type=str,
        default="generation/depth_maps",
        help="Directory to save depth maps",
    )
    parser.add_argument(
        "--foundation_pose_input_dir",
        type=str,
        default="generation/foundation_pose",
        help="Directory containing foundation pose input data",
    )

    # Render settings
    parser.add_argument("--samples", type=int, default=32, help="Number of samples (default: 32)")
    parser.add_argument("--width", type=int, default=WIDTH, help=f"Render width (default: {WIDTH})")
    parser.add_argument(
        "--height", type=int, default=HEIGHT, help=f"Render height (default: {HEIGHT})"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU rendering (default: True)",
    )

    # Initial state options
    parser.add_argument(
        "--use_initial_state",
        action="store_true",
        default=True,
        help="Use simulated initial state orientation for object",
    )
    parser.add_argument(
        "--initial_state_dir",
        type=str,
        default="generation/initial_states",
        help="Directory containing initial state data",
    )
    # Object scale options
    parser.add_argument(
        "--use_obj_scale",
        action="store_true",
        default=True,
        help="Use object scale from scale determination step",
    )
    parser.add_argument(
        "--obj_scale_dir",
        type=str,
        default="generation/obj_scales",
        help="Directory containing object scale data",
    )
    parser.add_argument(
        "--use_table_scene",
        action="store_true",
        default=False,
        help="Use table scene (default: False)",
    )

    # Other options
    parser.add_argument("--skip_done", action="store_true", help="Skip already processed files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--phase", type=str, default=None, help="Phase of the scene (default: None)"
    )
    parser.add_argument(
        "--obj_scale_override",
        type=float,
        nargs=3,
        default=None,
        help="Override object scale [x, y, z]",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        default=False,
        help="Only render the RGB image and exit (skip depth, masks, camera data, etc.)",
    )
    parser.add_argument(
        "--scene_config",
        type=str,
        default=None,
        help="Path to scenes YAML config file (legacy, prefer --pipeline_config)",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default=None,
        help="Path to pipeline YAML config file (scenes are extracted from it)",
    )
    parser.add_argument(
        "--object_config",
        type=str,
        default=None,
        help="Path to object YAML config file (default: configs/objects/default.yaml)",
    )
    parser.add_argument(
        "--character_init_pose_file",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to pose file(s). If two paths are given, the second is used for phase=end. Supports: (1) simple .npz from save_renderpeople_pose(); (2) motion .npz with motion_global/poses",
    )

    # When invoked via `blender --background --python <script> -- <args>`,
    # sys.argv contains Blender's own args before "--".  Strip them so
    # argparse only sees the script's arguments.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    if args.character_usd and (not args.character_name or args.character_name == "G1"):
        parser.error("--character_usd requires a non-G1 --character_name output identifier")
    if args.character_scale is not None and any(
        not math.isfinite(value) or value <= 0 for value in args.character_scale
    ):
        parser.error("--character_scale must contain three finite positive numbers")

    if args.scene_config:
        SCENE_CONFIG = load_scene_config(args.scene_config)
    elif args.pipeline_config:
        pcfg = load_pipeline_config(args.pipeline_config)
        SCENE_CONFIG = pcfg.get("scenes", {})
    else:
        SCENE_CONFIG = load_scene_config()
    OBJECT_CONFIG = load_object_config(args.object_config)

    try:
        # Determine object configuration
        category_prefix = "_".join(
            args.category.split("_")[:-1]
        )  # special case for category like "canned_food_3" from RoboCasa
        obj_config = None
        if args.category in OBJECT_CONFIG or category_prefix in OBJECT_CONFIG:
            obj_config = (
                OBJECT_CONFIG[args.category]
                if args.category in OBJECT_CONFIG
                else OBJECT_CONFIG[category_prefix]
            )
            print("Loaded obj_config: ", obj_config)

        # Determine scene configuration
        if args.scene is not None:
            scene_key = args.scene
            print(f"Using scene '{scene_key}' from command line")
        elif obj_config is not None and "scene" in obj_config:
            scene_key = obj_config["scene"]
            print(f"Auto-selected scene '{scene_key}' for category '{args.category}'")
        else:
            if args.use_table_scene:
                scene_key = OBJECT_CONFIG["table-default"]["scene"]
            else:
                scene_key = OBJECT_CONFIG["floor-default"]["scene"]
            print(f"No scene specified for category '{args.category}', using '{scene_key}'")

        if scene_key not in SCENE_CONFIG:
            raise KeyError(
                f"Scene '{scene_key}' not found in scene config "
                f"(available: {sorted(SCENE_CONFIG.keys())})"
            )

        scene_config = SCENE_CONFIG[scene_key]
        print(f"Using scene configuration: {scene_key}")
        print(f"Scene config: {scene_config}")

        if "blender_file_name" in scene_config:
            blender_file_name = scene_config["blender_file_name"]
        else:
            blender_file_name = scene_key
        blender_file = scene2blender(blender_file_name)

        # Create output directories
        os.makedirs(f"{args.results_dir}/{args.image_save_dir}", exist_ok=True)
        os.makedirs(f"{args.results_dir}/{args.camera_dir}", exist_ok=True)

        # new_category = args.category.replace(' ', '-').replace('_', '-')
        category = args.category
        # Create FoundationPose input directories
        parent_dir = os.path.dirname(f"{args.results_dir}/{args.foundation_pose_input_dir}")
        foundation_pose_mesh_dir = f"{parent_dir}/mesh/{args.dataset}/{category}"
        print(f"FoundationPose mesh directory: {foundation_pose_mesh_dir}")

        os.makedirs(foundation_pose_mesh_dir, exist_ok=True)

        # scene_id = f"{args.dataset}_{new_category}_{args.character_name if args.character_name else 'wo_character'}_{scene_key}"
        scene_id = f"{args.dataset}/{category}/{args.character_name if args.character_name else 'wo_character'}_{scene_key}"
        if args.rand_scene_seed is not None:
            if args.phase is None:
                scene_id += f"_rand{args.rand_scene_seed:05d}"
            else:
                scene_id += f"_{args.phase}{args.rand_scene_seed:05d}"
            combined_seed = int(
                hashlib.md5(
                    f"{args.base_seed}_{category}_{args.rand_scene_seed}".encode()
                ).hexdigest(),
                16,
            ) % (2**31)

        output_path = f"{args.results_dir}/{args.image_save_dir}/{scene_id}.png"
        camera_save_path = f"{args.results_dir}/{args.camera_dir}/{scene_id}.pickle"

        # FoundationPose input paths
        foundation_pose_intrinsics_path = (
            f"{args.results_dir}/{args.foundation_pose_input_dir}/{scene_id}/cam_K.txt"
        )
        foundation_pose_mask_path = (
            f"{args.results_dir}/{args.foundation_pose_input_dir}/{scene_id}/masks/000000.png"
        )
        foundation_pose_human_mask_path = (
            f"{args.results_dir}/{args.foundation_pose_input_dir}/{scene_id}/human_masks/000000.png"
        )
        foundation_pose_camera_path = f"{args.results_dir}/{args.foundation_pose_input_dir}/{scene_id}/first_frame_pose.pickle"

        # Check if already done
        foundation_pose_files = [
            foundation_pose_intrinsics_path,
            foundation_pose_mask_path,
            foundation_pose_human_mask_path,
            foundation_pose_camera_path,
        ]
        if args.skip_done:
            if args.phase == "end":
                if os.path.exists(output_path):
                    print(f"Already processed (end phase): {output_path}")
                    return
            elif (
                os.path.exists(output_path)
                and os.path.exists(camera_save_path)
                and all(os.path.exists(f) for f in foundation_pose_files)
            ):
                print(f"Already processed: {output_path}")
                return

        # Load blend file if provided, otherwise start with empty scene
        if blender_file:
            load_blend_file(blender_file)

        # Override scene config with object-specific config if available.
        # Whitelist of fields the per-object yaml is allowed to override on the scene.
        _OBJ_CONFIG_OVERRIDES = (
            "obj_rot",
            "obj_pos",
            "obj_scale",
            "obj_opacity",
            "obj_rot_rand_max",
            "character_pos",
            "character_rot",
            "character_rot_rand_max",
            "character_scale",
            "camera_radius",
            "camera_azimuth",
            "camera_elevation",
            "camera_offset",
        )
        if obj_config is not None:
            for key in _OBJ_CONFIG_OVERRIDES:
                if key in obj_config:
                    scene_config[key] = obj_config[key]
                    print(f"Overriding {key} from OBJECT_CONFIG: {obj_config[key]}")

        # Load obj_scale from scale determination step (similar to use_initial_state)
        if args.use_obj_scale:
            obj_scale_data = load_obj_scale_data(
                args.dataset, args.category, f"{args.results_dir}/{args.obj_scale_dir}"
            )
            if obj_scale_data is not None:
                scene_config["obj_scale"] = obj_scale_data
                print(f"Using determined obj_scale: {obj_scale_data}")
            else:
                print("Failed to load obj_scale data, using default scale from config")

        # Override obj_scale from CLI if provided (takes highest priority)
        if args.obj_scale_override is not None:
            scene_config["obj_scale"] = list(args.obj_scale_override)
            print(f"Overriding obj_scale from CLI: {args.obj_scale_override}")

        # Determine object rotation
        obj_rot = scene_config.get("obj_rot", None)
        if args.use_initial_state:
            # Load initial state data from simulation
            initial_state_data = load_initial_state_data(
                args.dataset, args.category, f"{args.results_dir}/{args.initial_state_dir}"
            )
            if initial_state_data is not None:
                # Use quaternion from simulation and convert to euler angles
                rotation_radians = quaternion_to_euler_blender(
                    initial_state_data["obj_R_quat"], obj_rot
                )
                print(
                    f"Using simulated orientation: {[math.degrees(r) for r in rotation_radians]} degrees"
                )
            else:
                print("Failed to load initial state data, using default rotation from scene config")
                rotation_radians = [math.radians(deg) for deg in obj_rot]
        else:
            # Use rotation from scene config (original behavior)
            rotation_radians = [math.radians(deg) for deg in obj_rot]
            print(f"Using scene config rotation: {obj_rot} degrees")

        loaded_obj = load_object_from_category(
            args.dataset,
            args.category,
            position=scene_config["obj_pos"],
            rotation=rotation_radians,
            scale=scene_config["obj_scale"],
        )

        # Print object dimensions
        print_object_dimensions(loaded_obj)

        # Apply object opacity if specified in scene config
        if "obj_opacity" in scene_config:
            obj_opacity = scene_config["obj_opacity"]
            print(f"Applying object opacity: {obj_opacity}")
            set_object_opacity(loaded_obj, obj_opacity)

        # Load static objects from scene config if present
        loaded_static_objects = {}
        if "static_objects" in scene_config:
            print("Loading static objects from scene config...")
            loaded_static_objects = load_static_objects(scene_config["static_objects"])
            print(f"Loaded {len(loaded_static_objects)} static object(s)")

            for obj_key, obj in loaded_static_objects.items():
                print_object_dimensions(obj)

        if args.character_name is not None:
            character_scale = args.character_scale or scene_config["character_scale"]
            rand_seed = args.rand_scene_seed
            if scene_config.get("character_pos_mode") == "polar":
                character_position, character_rotation = compute_polar_character_placement(
                    scene_config,
                    scene_config["obj_pos"],
                    rand_seed,
                    set_seed=not args.no_rand_seed,
                )
            else:
                character_position = scene_config["character_pos"]
                character_rotation = [math.radians(deg) for deg in scene_config["character_rot"]]

            if args.character_name == "G1":
                # Load G1 robot from MuJoCo XML at rest pose
                loaded_character = load_g1_robot(
                    xml_path=args.g1_xml_path,
                    position=character_position,
                    rotation=character_rotation,
                )
            else:
                # Load the selected rigged character. DHGen assets use centimetres,
                # so the public G1 profile supplies scale 0.01 explicitly.
                loaded_character = load_renderpeople_from_name(
                    args.character_name,
                    usd_file_path=args.character_usd,
                    position=character_position,
                    rotation=character_rotation,
                    scale=character_scale,
                    # rand_seed=args.rand_scene_seed,
                    character_dir=args.character_dir,
                    texture_dir=args.texture_dir,
                )

                if args.character_init_pose_file is not None:
                    if len(args.character_init_pose_file) > 1 and args.phase == "end":
                        character_init_pose_file = args.character_init_pose_file[1]
                    else:
                        character_init_pose_file = args.character_init_pose_file[0]
                else:
                    character_init_pose_file = None

            # Update scene
            bpy.context.view_layer.update()
        else:
            loaded_character = None

        # Setup GPU rendering if requested
        if args.gpu:
            gpu_success = setup_gpu_rendering()
            if not gpu_success:
                print("GPU setup failed, continuing with CPU rendering")

        # Configure render settings
        if "render_width" in scene_config:
            args.width = scene_config["render_width"]
        if "render_height" in scene_config:
            args.height = scene_config["render_height"]
        configure_render_settings(
            output_path,
            samples=args.samples,
            resolution_x=args.width,
            resolution_y=args.height,
        )

        if not args.no_rand_seed and args.rand_scene_seed is not None:
            random.seed(combined_seed)

        # Setup camera with object targeting using scene config
        if args.rand_scene_seed is not None and "camera_azimuth_rand_max" in scene_config:
            if isinstance(scene_config["camera_azimuth_rand_max"], list):
                yaw_offset_deg = random.uniform(
                    scene_config["camera_azimuth_rand_max"][0],
                    scene_config["camera_azimuth_rand_max"][1],
                )
            else:
                yaw_offset_deg = random.uniform(
                    -scene_config["camera_azimuth_rand_max"],
                    scene_config["camera_azimuth_rand_max"],
                )
        else:
            yaw_offset_deg = 0.0

        camera = setup_camera(
            width=args.width,
            height=args.height,
            senser_size=10,
            target_object=loaded_obj,
            target_character=loaded_character,
            elevation=scene_config["camera_elevation"],
            azimuth=scene_config["camera_azimuth"] + yaw_offset_deg,
            radius=scene_config["camera_radius"],
            camera_offset=scene_config["camera_offset"],
        )

        # adjust character/object position after camera setup to preserve the same camera for the same scene
        # camera_azimuth_rand_max should not be used
        if args.phase != "end":
            if "character_pos_start_offset" in scene_config:
                for i in range(3):
                    loaded_character.location[i] += scene_config["character_pos_start_offset"][i]
            if "obj_pos_start_offset" in scene_config:
                for i in range(3):
                    loaded_obj.location[i] += scene_config["obj_pos_start_offset"][i]
        elif args.phase == "end":
            if "character_pos_end_offset" in scene_config:
                for i in range(3):
                    loaded_character.location[i] += scene_config["character_pos_end_offset"][i]
            if "obj_pos_end_offset" in scene_config:
                for i in range(3):
                    loaded_obj.location[i] += scene_config["obj_pos_end_offset"][i]

        # Randomize character/object position for diversity, with penetration retry loop
        if args.rand_scene_seed is not None:

            # randomize table height
            table_offset = [0, 0, 0]
            if "static_objects" in scene_config:
                assert (
                    len(loaded_static_objects) == 1
                ), "Only one static object is supported for now"
                for obj_key, obj in loaded_static_objects.items():
                    if "pos_rand_max" in scene_config["static_objects"][obj_key]:
                        offsets = sample_rand_3d(
                            scene_config["static_objects"][obj_key], "pos_rand_max", "pos_rand_min"
                        )
                        for i in range(3):
                            table_offset[i] += offsets[i]
                            obj.location[i] += table_offset[i]
                            if obj.get("obj_on_table", True):
                                loaded_obj.location[i] += table_offset[i]

            # Save initial transforms so we can restore them on each retry
            char_loc_init = [loaded_character.location[i] for i in range(3)]
            char_rot_init = [loaded_character.rotation_euler[i] for i in range(3)]
            obj_loc_init = [loaded_obj.location[i] for i in range(3)]
            obj_rot_init = [loaded_obj.rotation_euler[i] for i in range(3)]
            static_locs_init = {
                key: [obj.location[i] for i in range(3)]
                for key, obj in loaded_static_objects.items()
            }

            max_penetration_retries = 100
            for attempt in range(max_penetration_retries):
                # Restore all transforms to pre-randomization state
                for i in range(3):
                    loaded_character.location[i] = char_loc_init[i]
                    loaded_character.rotation_euler[i] = char_rot_init[i]
                    loaded_obj.location[i] = obj_loc_init[i]
                    loaded_obj.rotation_euler[i] = obj_rot_init[i]
                for key, obj in loaded_static_objects.items():
                    for i in range(3):
                        obj.location[i] = static_locs_init[key][i]

                if character_init_pose_file is not None:
                    repose_renderpeople_from_saved(
                        loaded_character, character_init_pose_file, ground_z=character_position[2]
                    )

                if args.phase != "end":
                    if "character_pos_box_include" in scene_config:
                        dx, dy = sample_xy_in_box(
                            scene_config["character_pos_box_include"],
                            scene_config.get("character_pos_box_exclude", None),
                        )
                        loaded_character.location[0] += dx
                        loaded_character.location[1] += dy
                    elif (
                        "character_pos_rand_max" in scene_config
                        and scene_config.get("character_pos_mode") != "polar"
                    ):
                        offsets = sample_rand_3d(
                            scene_config, "character_pos_rand_max", "character_pos_rand_min"
                        )
                        for i in range(3):
                            loaded_character.location[i] += offsets[i]
                    if "character_rot_rand_max" in scene_config:
                        offsets = sample_rand_3d(
                            scene_config, "character_rot_rand_max", "character_rot_rand_min"
                        )
                        for i in range(3):
                            loaded_character.rotation_euler[i] += math.radians(offsets[i])

                if "obj_pos_rand_max" in scene_config:
                    offsets = sample_rand_3d(scene_config, "obj_pos_rand_max", "obj_pos_rand_min")
                    for i in range(3):
                        loaded_obj.location[i] += offsets[i]
                    print(f"Object position randomized: {loaded_obj.location}")
                if "obj_rot_rand_max" in scene_config:
                    offsets = sample_rand_3d(scene_config, "obj_rot_rand_max", "obj_rot_rand_min")
                    for i in range(3):
                        loaded_obj.rotation_euler[i] += math.radians(offsets[i])
                    print(
                        f"Object rotation randomized: {[math.degrees(loaded_obj.rotation_euler[i]) for i in range(3)]}"
                    )

                bpy.context.view_layer.update()

                if not check_character_penetration(
                    loaded_character, loaded_static_objects, loaded_obj
                ):
                    print(f"No penetration detected (attempt {attempt + 1})")
                    break
                print(
                    f"Penetration detected (attempt {attempt + 1}/{max_penetration_retries}), retrying..."
                )
            else:
                print(
                    f"Warning: penetration persists after {max_penetration_retries} attempts, exiting..."
                )
                exit(1)

        if args.render_only or args.phase == "end":
            render_scene()
            print(f"Render saved to: {output_path}")
            print("Skipping depth, masks, camera data, etc.")
            return

        # Calculate and save camera intrinsics for FoundationPose
        intrinsics = get_camera_intrinsics(camera, args.width, args.height)
        print_camera_intrinsics(intrinsics)
        save_camera_intrinsics(intrinsics, foundation_pose_intrinsics_path)

        # Copy mesh files for FoundationPose
        obj_file_path = category2object(f"data/{args.dataset}", args.category)
        copy_mesh_files(obj_file_path, foundation_pose_mesh_dir)

        # Save camera and object data
        save_camera_and_object_data(
            camera=camera,
            obj=loaded_obj,
            camera_save_path=camera_save_path,
            focal_length=FOCAL_LENGTH,
            resolution=(args.width, args.height),
        )

        # Save FoundationPose-compatible camera data
        save_foundation_pose_camera_data(
            camera=camera,
            obj=loaded_obj,
            character=loaded_character,
            static_objects=loaded_static_objects,
            support_object_names=scene_config.get("support_object_names"),
            frame_height=args.height,
            frame_width=args.width,
            focal_length=FOCAL_LENGTH,
            foundation_pose_camera_save_path=foundation_pose_camera_path,
        )

        # Save character metadata (height, scale, etc.)
        if loaded_character is not None:
            character_data_path = f"{args.results_dir}/{args.foundation_pose_input_dir}/{scene_id}/character_data.pickle"
            save_character_data(loaded_character, args.character_name, character_data_path)

        # Render the scene with depth map
        depth_output_path = f"{args.results_dir}/{args.depth_save_dir}/{scene_id}"
        os.makedirs(depth_output_path, exist_ok=True)
        depth_output_path = f"{depth_output_path}/000000"
        render_scene(depth_output_path=depth_output_path)
        print(f"Render saved to: {output_path}")

        # Render object mask for FoundationPose
        render_object_mask(loaded_obj, foundation_pose_mask_path)

        # Render human/character mask if character is present
        if loaded_character is not None:
            render_character_mask(loaded_character, foundation_pose_human_mask_path)

        print(f"Render saved to: {output_path}")
        print(f"Depth map saved to: {depth_output_path}.png")
        print(f"Camera data saved to: {camera_save_path}")
        print("FoundationPose data saved:")
        print(f"  Intrinsics: {foundation_pose_intrinsics_path}")
        print(f"  Mask: {foundation_pose_mask_path}")
        print(f"  Camera data: {foundation_pose_camera_path}")
        print(f"  Mesh files: {foundation_pose_mesh_dir}")

    except Exception as e:
        import traceback

        print(f"Error during rendering: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
