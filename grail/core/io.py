import hashlib
import os
import pickle
import subprocess
import time
from glob import glob

import imageio
import numpy as np
from tqdm import tqdm

from .video import normalize_motion_fps

try:
    import torch

    from .torch_utils import tensor_to
except ImportError:
    pass


def vis_keypoints_data(video_path, vitpose, hand_keypoints_2d, cache_dir):
    """
    Visualize keypoints data on video frames.

    Supports both COCO17 (17 joints) and SOMA77 (77 joints) vitpose formats.

    Args:
        video_path (str): Path to input video
        vitpose (np.ndarray or torch.Tensor): Body pose keypoints —
            (F, 17, 3) COCO17 format or (F, 77, 3) SOMA77 format, with (x, y, confidence)
        hand_keypoints_2d (dict or np.ndarray): Hand keypoints for left and right hands
        cache_dir (str): Directory to save the output visualization

    Returns:
        str: Path to the output video
    """
    import cv2

    # Convert torch tensors to numpy if needed
    if vitpose is not None:
        if hasattr(vitpose, "cpu"):
            vitpose = vitpose.cpu().numpy()
        vitpose = np.array(vitpose)

    if hand_keypoints_2d is not None and not isinstance(hand_keypoints_2d, dict):
        if hasattr(hand_keypoints_2d, "cpu"):
            hand_keypoints_2d = hand_keypoints_2d.cpu().numpy()
        hand_keypoints_2d = np.array(hand_keypoints_2d)

    # Read video frames
    reader = imageio.get_reader(video_path)
    fps = normalize_motion_fps(reader.get_meta_data().get("fps"))
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()

    num_frames = len(frames)
    print(f"Processing {num_frames} frames from video: {video_path}")

    # COCO17 skeleton connections
    coco_skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]

    # SOMA77 parent indices (child -> parent)
    soma77_parents = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        6,
        3,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        14,
        19,
        20,
        21,
        22,
        14,
        24,
        25,
        26,
        27,
        14,
        29,
        30,
        31,
        32,
        14,
        34,
        35,
        36,
        37,
        3,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        42,
        47,
        48,
        49,
        50,
        42,
        52,
        53,
        54,
        55,
        42,
        57,
        58,
        59,
        60,
        42,
        62,
        63,
        64,
        65,
        0,
        67,
        68,
        69,
        70,
        0,
        72,
        73,
        74,
        75,
    ]

    # SOMA77 body-part color mapping (RGB for imageio frames)
    soma77_joint_group = [""] * 77
    for j in range(0, 4):
        soma77_joint_group[j] = "torso"
    for j in range(4, 11):
        soma77_joint_group[j] = "head"
    for j in range(11, 14):
        soma77_joint_group[j] = "left_arm"
    for j in range(14, 39):
        soma77_joint_group[j] = "left_hand"
    for j in range(39, 42):
        soma77_joint_group[j] = "right_arm"
    for j in range(42, 67):
        soma77_joint_group[j] = "right_hand"
    for j in range(67, 72):
        soma77_joint_group[j] = "left_leg"
    for j in range(72, 77):
        soma77_joint_group[j] = "right_leg"

    soma77_colors = {
        "torso": (255, 215, 0),
        "head": (255, 130, 180),
        "left_arm": (100, 255, 0),
        "left_hand": (130, 255, 130),
        "right_arm": (255, 130, 50),
        "right_hand": (255, 180, 130),
        "left_leg": (0, 190, 255),
        "right_leg": (170, 0, 255),
    }

    # Hand skeleton connections (21 keypoints per hand)
    hand_skeleton = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],  # Thumb
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],  # Index
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],  # Middle
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],  # Ring
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],  # Pinky
    ]

    # Detect vitpose format
    is_soma77 = vitpose is not None and vitpose.shape[1] == 77

    output_frames = []

    for i in tqdm(range(num_frames), desc="Drawing keypoints"):
        frame = frames[i].copy()

        if vitpose is not None and i < len(vitpose):
            body_kpts = vitpose[i]
            conf_thr = 0.3

            if is_soma77:
                # SOMA77: draw bones using parent hierarchy with body-part colors
                for child_idx, parent_idx in enumerate(soma77_parents):
                    if parent_idx < 0:
                        continue
                    kp_c = body_kpts[child_idx]
                    kp_p = body_kpts[parent_idx]
                    if len(kp_c) >= 3 and len(kp_p) >= 3:
                        if kp_c[2] <= conf_thr or kp_p[2] <= conf_thr:
                            continue
                    pt1 = (int(kp_p[0]), int(kp_p[1]))
                    pt2 = (int(kp_c[0]), int(kp_c[1]))
                    group = soma77_joint_group[child_idx]
                    color = soma77_colors.get(group, (0, 255, 0))
                    thickness = 2 if "hand" in group else 3
                    cv2.line(frame, pt1, pt2, color, thickness)

                for j, kp in enumerate(body_kpts):
                    if len(kp) >= 3 and kp[2] <= conf_thr:
                        continue
                    pt = (int(kp[0]), int(kp[1]))
                    group = soma77_joint_group[j]
                    color = soma77_colors.get(group, (0, 0, 255))
                    radius = 3 if "hand" in group else 5
                    cv2.circle(frame, pt, radius, color, -1)
            else:
                # COCO17: original skeleton drawing
                for bone in coco_skeleton:
                    if bone[0] < len(body_kpts) and bone[1] < len(body_kpts):
                        kp1 = body_kpts[bone[0]]
                        kp2 = body_kpts[bone[1]]

                        if len(kp1) >= 3 and len(kp2) >= 3:
                            if kp1[2] > conf_thr and kp2[2] > conf_thr:
                                pt1 = (int(kp1[0]), int(kp1[1]))
                                pt2 = (int(kp2[0]), int(kp2[1]))
                                cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                        else:
                            pt1 = (int(kp1[0]), int(kp1[1]))
                            pt2 = (int(kp2[0]), int(kp2[1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

                for j, kp in enumerate(body_kpts):
                    if len(kp) >= 3:
                        if kp[2] > conf_thr:
                            pt = (int(kp[0]), int(kp[1]))
                            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                    else:
                        pt = (int(kp[0]), int(kp[1]))
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        # Draw separate hand detections when supplied.
        if hand_keypoints_2d is not None and not is_soma77:
            if isinstance(hand_keypoints_2d, dict):
                # Handle dict format with 'left' and 'right' keys
                for hand_name, hand_color in [("left", (255, 0, 0)), ("right", (0, 165, 255))]:
                    if hand_name in hand_keypoints_2d and i < len(hand_keypoints_2d[hand_name]):
                        hand_kpts = hand_keypoints_2d[hand_name][i]
                        if hand_kpts is not None and len(hand_kpts) > 0:
                            # Draw hand skeleton
                            for bone in hand_skeleton:
                                if bone[0] < len(hand_kpts) and bone[1] < len(hand_kpts):
                                    kp1 = hand_kpts[bone[0]]
                                    kp2 = hand_kpts[bone[1]]

                                    if len(kp1) >= 3 and len(kp2) >= 3:  # Has confidence
                                        if kp1[2] > 0.2 and kp2[2] > 0.2:
                                            pt1 = (int(kp1[0]), int(kp1[1]))
                                            pt2 = (int(kp2[0]), int(kp2[1]))
                                            cv2.line(frame, pt1, pt2, hand_color, 2)
                                    else:
                                        # No confidence scores, draw all connections
                                        pt1 = (int(kp1[0]), int(kp1[1]))
                                        pt2 = (int(kp2[0]), int(kp2[1]))
                                        cv2.line(frame, pt1, pt2, hand_color, 2)

                            # Draw hand keypoints with index labels
                            for kp_idx, kp in enumerate(hand_kpts):
                                if len(kp) >= 3:  # Has confidence
                                    if kp[2] > 0.2:
                                        pt = (int(kp[0]), int(kp[1]))
                                        cv2.circle(frame, pt, 3, hand_color, -1)
                                        # Draw index number
                                        cv2.putText(
                                            frame,
                                            str(kp_idx),
                                            (pt[0] + 5, pt[1] - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4,
                                            hand_color,
                                            1,
                                            cv2.LINE_AA,
                                        )
                                else:
                                    # No confidence scores, draw all keypoints
                                    pt = (int(kp[0]), int(kp[1]))
                                    cv2.circle(frame, pt, 3, hand_color, -1)
                                    # Draw index number
                                    cv2.putText(
                                        frame,
                                        str(kp_idx),
                                        (pt[0] + 5, pt[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        hand_color,
                                        1,
                                        cv2.LINE_AA,
                                    )
            elif hasattr(hand_keypoints_2d, "__len__") and i < len(hand_keypoints_2d):
                # Handle array format (all hands together)
                # First half (0:16) = left hand, second half (16:32) = right hand
                hand_kpts = hand_keypoints_2d[i]
                if hand_kpts is not None and len(hand_kpts) > 0:
                    num_kpts = len(hand_kpts)
                    half = num_kpts // 2  # 16 for left, 16 for right
                    for kp_idx, kp in enumerate(hand_kpts):
                        if len(kp) >= 2:
                            conf = kp[2] if len(kp) >= 3 else 1.0
                            if conf > 0.2:
                                pt = (int(kp[0]), int(kp[1]))
                                # Left hand (first half): green, Right hand (second half): magenta
                                if kp_idx < half:
                                    color = (0, 255, 0)  # green for left
                                    text_color = (0, 200, 0)
                                else:
                                    color = (255, 0, 255)  # magenta for right
                                    text_color = (200, 0, 200)
                                cv2.circle(frame, pt, 3, color, -1)
                                # Draw index number
                                cv2.putText(
                                    frame,
                                    str(kp_idx),
                                    (pt[0] + 5, pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    text_color,
                                    1,
                                    cv2.LINE_AA,
                                )

        output_frames.append(frame)

    # Save output video
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, "keypoints_visualization.mp4")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in output_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved keypoints visualization to: {output_path}")
    return output_path


def save_human_motion_data(motion_global, motion_incam, output_path):
    """
    Save SMPL motion data to a file

    Args:
        motion_global (dict): Global motion data
        motion_incam (dict): In-camera motion data
        output_path (str): Path to save the data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, motion_global=motion_global, motion_incam=motion_incam)


def load_human_motion_data(
    hmr_file_path, model="soma03", is_global=False, to_tensor=False, device="cuda"
):
    """
    Load native SOMA motion data from a file

    Args:
        hmr_file_path (str): Path to the .npz file containing motion data
    Returns:
        dict: Dictionary containing motion data
    """

    if model not in ("soma03", "soma"):
        raise ValueError(f"Unsupported human motion model: {model!r}")
    if not os.path.exists(hmr_file_path):
        raise FileNotFoundError(f"SOMA motion file not found: {hmr_file_path}")
    with np.load(hmr_file_path, allow_pickle=True) as archive:
        motion_data = archive["motion_global" if is_global else "motion_incam"].item()

    if to_tensor:
        motion_data = tensor_to(motion_data, device)

    return motion_data


def save_object_pose_data(pose_list, save_path):
    """
    Save object pose data to a file

    Args:
        pose_list (list): List of 4x4 transformation matrices for each frame
        save_path (str): Path to save the data
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(pose_list, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_object_pose_data(pose_file_path, to_tensor=False, device="cuda"):
    """
    Load object pose data from foundation_pose tracking results

    Args:
        pose_file_path (str): Path to poses_in_cam.pkl file

    Returns:
        list: List of 4x4 transformation matrices for each frame
    """
    if not os.path.exists(pose_file_path):
        raise FileNotFoundError(f"Object pose file not found: {pose_file_path}")

    with open(pose_file_path, "rb") as f:
        pose_list = pickle.load(f)

    print(f"Loaded object pose data: {len(pose_list)} frames")

    if to_tensor:
        pose_list = np.array(pose_list).astype(np.float32)
        pose_list = torch.from_numpy(pose_list).to(device)

    return pose_list


def save_hoi_data(hoi_data, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(hoi_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_hoi_data(load_path, output_eval_data=False):
    with open(load_path, "rb") as f:
        hoi_data = pickle.load(f)
    return hoi_data

    # if output_eval_data:
    #     return hoi_data["human_data"], hoi_data["obj_data"], hoi_data.get("meta", {}), hoi_data["eval_data"]
    # else:
    #     return hoi_data["human_data"], hoi_data["obj_data"], hoi_data.get("meta", {})


def save_init_sim_state_data():
    # TODO: Implement this function
    pass


def load_init_sim_state_data():
    # TODO: Implement this function
    pass


def load_init_rendering_data(
    load_path, to_tensor=False, with_human_data=False, with_scene_data=False, device="cuda"
):
    with open(load_path, "rb") as handle:
        init_rendering_data = pickle.load(handle)

    obj_R = init_rendering_data["obj_R"]
    obj_t = init_rendering_data["obj_t"].reshape((3,))
    obj_scale = init_rendering_data["obj_scale"]

    cam_R = init_rendering_data["cam_R"]
    cam_t = init_rendering_data["cam_t"].reshape((3,))

    if "focal_length" in init_rendering_data:
        render_config = (
            init_rendering_data["frame_height"],
            init_rendering_data["frame_width"],
            init_rendering_data["focal_length"],
        )
    else:
        render_config = None

    if to_tensor:
        obj_R = torch.from_numpy(obj_R).float().to(device)
        obj_t = torch.from_numpy(obj_t).float().to(device)
        obj_scale = torch.from_numpy(obj_scale).float().to(device)
        cam_R = torch.from_numpy(cam_R).float().to(device)
        cam_t = torch.from_numpy(cam_t).float().to(device)

    additional_data = {}
    if with_human_data:
        human_R = init_rendering_data.get("character_R", None)
        human_t = init_rendering_data.get("character_t", None)
        if human_t is not None:
            human_t = human_t.reshape((3,))
            if to_tensor:
                human_t = torch.from_numpy(human_t).float().to(device)
                human_R = torch.from_numpy(human_R).float().to(device)
        additional_data["human_R"] = human_R
        additional_data["human_t"] = human_t
    if with_scene_data:
        if "static_objects" in init_rendering_data:
            additional_data["static_objects"] = init_rendering_data["static_objects"]
            if to_tensor:
                for obj_data in additional_data["static_objects"].values():
                    # Convert geometry, preserving integer topology and nested provenance.
                    for key in ("scale", "rot", "pos", "vertices"):
                        if key in obj_data:
                            obj_data[key] = torch.as_tensor(
                                obj_data[key], dtype=torch.float32, device=device
                            )
                    if "faces" in obj_data:
                        obj_data["faces"] = torch.as_tensor(
                            obj_data["faces"], dtype=torch.int64, device=device
                        )

    if with_human_data or with_scene_data:
        return obj_R, obj_t, obj_scale, cam_R, cam_t, render_config, additional_data
    else:
        return obj_R, obj_t, obj_scale, cam_R, cam_t, render_config


def serialize_support_objects(support_objects, source_blend_path):
    """Embed explicitly selected, unmodified Blender meshes in world coordinates.

    No evaluated mesh or transform decomposition is used: the original vertices
    and Blender loop triangles are transformed by the full matrix_world, including
    any parent transform or shear. The source file and original matrix remain
    available for provenance. This helper does not change scene placement.
    """
    if not support_objects:
        return {}
    if not source_blend_path or not os.path.isfile(source_blend_path):
        raise ValueError("Selected support objects require an existing source .blend file")
    source_blend_path = os.path.realpath(source_blend_path)
    source_hash = hashlib.sha256()
    with open(source_blend_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            source_hash.update(chunk)

    serialized = {}
    for name, obj in support_objects.items():
        if obj is None or obj.name != name or obj.type != "MESH":
            raise ValueError(f"Selected support object {name!r} must be an existing named mesh")
        current = obj
        while current is not None:
            rigid_body = getattr(current, "rigid_body", None)
            if (
                getattr(current, "animation_data", None)
                or getattr(current, "constraints", ())
                or getattr(current, "modifiers", ())
                or (rigid_body is not None and rigid_body.type != "PASSIVE")
            ):
                raise ValueError(f"Support object {name!r} has non-static or modified ancestry")
            current = getattr(current, "parent", None)
        mesh = obj.data
        if (
            getattr(mesh, "animation_data", None)
            or getattr(mesh, "shape_keys", None)
            or getattr(obj, "is_instancer", False)
        ):
            raise ValueError(f"Support object {name!r} has animated or instanced geometry")
        matrix = np.array(obj.matrix_world, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError(f"Support object {name!r} has an invalid world transform")
        mesh.calc_loop_triangles()
        vertices = np.array([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        faces = np.array([triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ValueError(f"Support object {name!r} has invalid or empty mesh geometry")
        serialized[name] = {
            "vertices": vertices @ matrix[:3, :3].T + matrix[:3, 3],
            "faces": faces,
            "pos": np.zeros(3),
            "rot": np.eye(3),
            "scale": np.ones(3),
            "source_matrix_world": matrix,
            "source_blend": {"path": source_blend_path, "sha256": source_hash.hexdigest()},
            "source_object": name,
            "role": "support",
        }
    return serialized


def save_init_rendering_data(
    save_path,
    cam_R,
    cam_t,
    frame_height,
    frame_width,
    focal_length,
    obj_R,
    obj_t,
    obj_scale,
    character_R,
    character_t,
    character_scale,
    static_objects,
    support_objects=None,
    source_blend_path=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    init_rendering_data = dict(
        cam_R=np.array(cam_R).reshape((3, 3)),
        cam_t=np.array(cam_t).reshape((3, 1)),
        obj_R=np.array(obj_R).reshape((3, 3)),
        obj_t=np.array(obj_t).reshape((3, 1)),
        obj_scale=np.array(obj_scale).reshape((3, 1)),
        character_R=np.array(character_R).reshape((3, 3)),
        character_t=np.array(character_t).reshape((3, 1)),
        character_scale=np.array(character_scale).reshape((3, 1)),
        frame_height=frame_height,
        frame_width=frame_width,
        focal_length=focal_length,
    )
    if static_objects is not None:
        init_rendering_data["static_objects"] = {}
        for obj_key, obj in static_objects.items():
            init_rendering_data["static_objects"][obj_key] = {
                "path": obj["path"],
                "pos": np.array(obj.location).reshape((3,)),
                "rot": np.array(obj.rotation_euler.to_matrix()).reshape((3, 3)),
                "scale": np.array(obj.scale).reshape((3,)),
            }

    if support_objects:
        scene_objects = init_rendering_data.setdefault("static_objects", {})
        overlap = scene_objects.keys() & support_objects.keys()
        if overlap:
            raise ValueError(
                f"Support object keys duplicate imported static objects: {sorted(overlap)}"
            )
        scene_objects.update(serialize_support_objects(support_objects, source_blend_path))

    with open(save_path, "wb") as handle:
        pickle.dump(init_rendering_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_camera_intrinsics(intrinsics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    K = intrinsics["K"]

    with open(save_path, "w") as f:
        f.write(f"{K[0, 0]:.18e} {K[0, 1]:.18e} {K[0, 2]:.18e}\n")
        f.write(f"{K[1, 0]:.18e} {K[1, 1]:.18e} {K[1, 2]:.18e}\n")
        f.write(f"{K[2, 0]:.18e} {K[2, 1]:.18e} {K[2, 2]:.18e}\n")


def load_character_data(character_data_path):
    """Load character metadata (height, position, scale) from a pickle file.

    Returns None if the file doesn't exist.
    """
    if not os.path.exists(character_data_path):
        return None
    with open(character_data_path, "rb") as f:
        return pickle.load(f)


def load_mesh(mesh_path, mesh_scale=None, target_num_verts=None, device="cuda"):
    """
    Load a mesh from file with optional scaling and remeshing.

    Args:
        mesh_path: Path to mesh file
        mesh_scale: Scale factor (3D vector or scalar)
        target_num_verts: If specified, remesh to approximate this vertex count
        device: Device to load tensors on

    Returns:
        verts: Vertex positions (N, 3)
        faces: Face indices (F, 3)
        textures: Texture information (may be None)
    """
    import torch
    import trimesh
    from pytorch3d.io import load_objs_as_meshes

    from grail.rendering.textures import convert_textures_uv_to_vertex

    def remesh_to_target(trimesh_mesh, target_verts, tolerance=0.2):
        """
        Remesh to get close to target vertex count.

        Args:
            trimesh_mesh: trimesh.Trimesh object
            target_verts: Target number of vertices
            tolerance: Acceptable relative difference (e.g., 0.2 means within 20%)

        Returns:
            remeshed trimesh.Trimesh object
        """
        current_verts = len(trimesh_mesh.vertices)
        print(f"Current vertices: {current_verts}, Target: {target_verts}")

        # Check if already close enough
        relative_diff = abs(current_verts - target_verts) / target_verts
        if relative_diff < tolerance:
            print(f"Vertex count already within tolerance ({relative_diff:.1%}), skipping remesh")
            return trimesh_mesh

        if current_verts > target_verts:
            # Simplify mesh (reduce vertices)
            print(f"Simplifying mesh from {current_verts} to ~{target_verts} vertices...")
            # Estimate target face count (roughly 2*vertices for closed meshes)
            target_faces = int(target_verts * 2)
            try:
                # trimesh 4.x: first positional arg is `percent` (0-1); use
                # face_count kwarg to pass a target face count directly.
                trimesh_mesh = trimesh_mesh.simplify_quadric_decimation(face_count=target_faces)
                print(
                    f"Simplified to {len(trimesh_mesh.vertices)} vertices, {len(trimesh_mesh.faces)} faces"
                )
            except Exception as e:
                print(f"Warning: Simplification failed ({e}), using original mesh")
        else:
            # Subdivide mesh (increase vertices)
            print(f"Subdividing mesh from {current_verts} to ~{target_verts} vertices...")
            # Calculate how many subdivisions needed
            # Each subdivision roughly quadruples the face count
            ratio = target_verts / current_verts
            num_subdivisions = max(1, int(np.log(ratio) / np.log(4)))

            try:
                for i in range(num_subdivisions):
                    trimesh_mesh = trimesh_mesh.subdivide()
                    current = len(trimesh_mesh.vertices)
                    print(f"  Subdivision {i+1}/{num_subdivisions}: {current} vertices")
                    # Stop if we've exceeded target
                    if current >= target_verts * (1 + tolerance):
                        break

                # If we overshot, simplify back down
                current = len(trimesh_mesh.vertices)
                if current > target_verts * (1 + tolerance):
                    print(f"Overshot target, simplifying from {current} to ~{target_verts}...")
                    target_faces = int(target_verts * 2)
                    trimesh_mesh = trimesh_mesh.simplify_quadric_decimation(target_faces)
                    print(f"Final: {len(trimesh_mesh.vertices)} vertices")
            except Exception as e:
                print(f"Warning: Subdivision failed ({e}), using current mesh state")

        return trimesh_mesh

    # Load mesh using PyTorch3D
    try:
        meshes = load_objs_as_meshes([mesh_path], device=device)
        mesh = meshes[0]

        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        # Apply remeshing if target_num_verts is specified
        if target_num_verts is not None:
            print(f"Remeshing to target {target_num_verts} vertices...")
            # Convert to trimesh for remeshing
            trimesh_mesh = trimesh.Trimesh(
                vertices=verts.cpu().numpy(), faces=faces.cpu().numpy(), process=False
            )

            # Remesh
            trimesh_mesh = remesh_to_target(trimesh_mesh, target_num_verts)

            # Convert back to torch tensors
            verts = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.tensor(trimesh_mesh.faces, dtype=torch.long, device=device)

        # Handle both torch tensor and numpy array inputs for mesh_scale
        if torch.is_tensor(mesh_scale):
            scale_tensor = mesh_scale.to(device=device, dtype=torch.float32)
        else:
            scale_tensor = torch.tensor(mesh_scale, dtype=torch.float32, device=device)
        verts = verts * scale_tensor.reshape(1, 3)

        # Get textures (will be lost after remeshing)
        textures = None
        if target_num_verts is None:
            # Only preserve textures if no remeshing was done
            textures = convert_textures_uv_to_vertex(mesh, device)

        print(f"Loaded object mesh: {len(verts)} vertices, {len(faces)} faces")
        return verts, faces, textures

    except Exception as e:
        print(f"Error loading mesh with PyTorch3D: {e}")
        print("Falling back to basic vertex/face loading...")

        # Fallback: load without textures
        mesh = trimesh.load(mesh_path, force="mesh")

        # Apply remeshing if target_num_verts is specified
        if target_num_verts is not None:
            print(f"Remeshing to target {target_num_verts} vertices...")
            mesh = remesh_to_target(mesh, target_num_verts)

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

        # Apply object scaling to vertices
        # Check if scaling is needed (handle both tensor and numpy inputs)
        if torch.is_tensor(mesh_scale):
            is_unit_scale = torch.allclose(
                mesh_scale,
                torch.tensor([1.0, 1.0, 1.0], device=mesh_scale.device, dtype=mesh_scale.dtype),
            )
        else:
            is_unit_scale = np.allclose(mesh_scale, [1.0, 1.0, 1.0])

        if not is_unit_scale:
            # Handle both torch tensor and numpy array inputs for mesh_scale
            if torch.is_tensor(mesh_scale):
                scale_tensor = mesh_scale.to(device=device, dtype=torch.float32)
            else:
                scale_tensor = torch.tensor(mesh_scale, dtype=torch.float32, device=device)

            vertices = vertices * scale_tensor.reshape(1, 3)

        print(f"Loaded object mesh (fallback): {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces, None


def save_mesh(verts, faces, save_path):
    """
    Save mesh vertices and faces as OBJ file
    """
    import torch

    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            if len(face) == 3:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            elif len(face) == 4:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")


def run_subprocess(cmd, description, shell=False, env=None, cwd=None):
    """
    Run a subprocess with proper error handling and logging

    Args:
        cmd (list or str): Command to run
        description (str): Description for logging
        shell (bool): Whether to run in shell mode
        env (dict): Environment variables
        cwd (str): Working directory

    Returns:
        bool: True if successful, False otherwise
    """
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n🔄 {description}")
    print(f"Command: {cmd_str}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, shell=shell, check=True, capture_output=True, text=True, env=env, cwd=cwd
        )

        duration = time.time() - start_time
        print(f"✅ Completed in {duration:.2f}s")

        if result.stdout:
            print(result.stdout.strip())

        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Failed in {duration:.2f}s (code {e.returncode})")

        if e.stdout:
            print(f"STDOUT: {e.stdout.strip()}")
        if e.stderr:
            print(f"STDERR: {e.stderr.strip()}")

        return False

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Exception in {duration:.2f}s: {str(e)}")
        return False
