import os
import sys

import numpy as np
import torch
import trimesh

# Add parent directories to path
from grail.models.human_model import create_human_model


def post_process_hoi_result(hoi_data, mesh_data_dir, cfg):
    """
    Post-process HOI result by applying human_offset to human_data and obj_data,
    and computing table_surface_height.

    Args:
        hoi_data: Dictionary containing 'human_data', 'obj_data' and 'meta'
        mesh_data_dir: Directory for mesh data
        cfg: Filter config dict

    Returns:
        Modified hoi_data with updated human_data, obj_data, and table_surface_height
    """
    center_at_object = cfg["post_processing"].get("center_at_object", True)
    human_model_cfg = cfg["human_model"]
    device = "cpu"

    human_data = hoi_data["human_data"]
    obj_data = hoi_data["obj_data"]

    # Convert to torch tensors (CPU only)
    obj_t = torch.from_numpy(obj_data["obj_t"]).float()
    obj_R = torch.from_numpy(obj_data["obj_R"]).float()
    obj_scale = obj_data["obj_scale"].reshape(-1)

    human_trans = torch.from_numpy(human_data["trans"]).float()

    # ========== Compute human vertices ==========
    # Prepare motion data (supports both SMPLX and SOMA)
    motion_data = {
        "poses": torch.from_numpy(human_data["poses"]).float().to(device),
        "trans": torch.from_numpy(human_data["trans"]).float().to(device),
    }

    # SMPLX uses "betas", SOMA uses "identity_coeffs" + "scale_params"
    if "betas" in human_data:
        motion_data["betas"] = torch.from_numpy(human_data["betas"]).float().to(device)
    if "identity_coeffs" in human_data:
        motion_data["identity_coeffs"] = (
            torch.from_numpy(human_data["identity_coeffs"]).float().to(device)
        )
    if "scale_params" in human_data:
        motion_data["scale_params"] = (
            torch.from_numpy(human_data["scale_params"]).float().to(device)
        )

    if "left_hand_pose" in human_data:
        motion_data["left_hand_pose"] = (
            torch.from_numpy(human_data["left_hand_pose"]).float().to(device)
        )
    if "right_hand_pose" in human_data:
        motion_data["right_hand_pose"] = (
            torch.from_numpy(human_data["right_hand_pose"]).float().to(device)
        )
    motion_data["scale"] = human_data.get("scale", 1.0)

    frame_num = motion_data["poses"].shape[0]

    # Setup body model
    human_model = create_human_model(human_model_cfg, device=device)
    if human_model.model_type == "soma03":
        human_model.load_shape_params(human_model_cfg, None, device)
    if human_model.model_type == "g1_smplx":
        motion_data["scale"] = 1.0

    # Generate mesh with joints
    verts, faces, joints = human_model.generate_mesh(
        motion_data, output_joints=True, require_grad=False
    )
    if cfg["post_processing"].get("save_joints", False):
        hoi_data["human_data"]["joints"] = joints.cpu().numpy()

    # Get hand vertices for contact detection
    try:
        left_hand_verts = human_model.get_verts_segment(verts, ["L_Hand"])
        right_hand_verts = human_model.get_verts_segment(verts, ["R_Hand"])
    except Exception:
        from grail.models.smplx_model import get_smplx_verts_segment

        # Fallback: use SMPLX segment extraction
        left_hand_verts = get_smplx_verts_segment(verts, ["L_Hand"])
        right_hand_verts = get_smplx_verts_segment(verts, ["R_Hand"])

    # ========== Load object vertices (from process_data.py) ==========
    # Get object path from metadata
    obj_path = hoi_data["meta"]["obj_path"]
    obj_mesh = trimesh.load(obj_path, force="mesh")
    obj_vertices = np.asarray(obj_mesh.vertices)
    obj_vertices = obj_vertices * obj_scale
    obj_faces = np.asarray(obj_mesh.faces)

    # Find the contact points between the inter_start_idx and inter_end_idx
    inter_start_idx = hoi_data["meta"]["inter_start_idx"]
    inter_end_idx = hoi_data["meta"]["inter_end_idx"]

    contact_threshold = 0.05  # 5cm threshold for contact detection
    max_obj_verts = 10000  # Maximum object vertices for distance computation
    obj_contact_points = {}
    obj_verts_tensor = torch.from_numpy(obj_vertices).float()

    obj_contact_points_left_hand = {}
    obj_contact_points_right_hand = {}

    left_hand_inter_start_idx = -1
    right_hand_inter_start_idx = -1
    # for i in range(inter_start_idx, inter_end_idx):
    nframes_before = cfg["post_processing"].get(
        "compute_contact_points_nframes_before_inter_start", 48
    )
    for i in range(max(0, inter_start_idx - nframes_before), inter_end_idx):
        # Transform vertices to world space
        obj_verts_world = (obj_R[i] @ obj_verts_tensor.T).T + obj_t[i]
        left_hand_verts_world = left_hand_verts[i]
        right_hand_verts_world = right_hand_verts[i]

        # Downsample object vertices if needed to reduce memory usage
        if len(obj_verts_world) > max_obj_verts:
            downsample_indices = torch.randperm(len(obj_verts_world))[:max_obj_verts]
            obj_verts_sampled = obj_verts_world[downsample_indices]
        else:
            downsample_indices = torch.arange(len(obj_verts_world))
            obj_verts_sampled = obj_verts_world

        # Calculate contact points for LEFT hand
        diff_left = left_hand_verts_world.unsqueeze(1) - obj_verts_sampled.unsqueeze(0)
        distances_left = torch.norm(diff_left, dim=-1)
        min_dist_per_obj_left, _ = distances_left.min(dim=0)
        obj_contact_mask_left = min_dist_per_obj_left < contact_threshold
        obj_contact_indices_sampled_left = torch.where(obj_contact_mask_left)[0]
        obj_contact_indices_left = downsample_indices[obj_contact_indices_sampled_left]

        if len(obj_contact_indices_left) > 0:
            obj_contact_points_world_left = obj_verts_world[obj_contact_indices_left]
            obj_contact_points_local_left = (
                obj_R[i].T @ (obj_contact_points_world_left - obj_t[i]).T
            ).T
            obj_contact_points_left_hand[i] = obj_contact_points_local_left.numpy()
            if left_hand_inter_start_idx == -1:
                left_hand_inter_start_idx = i
        else:
            obj_contact_points_left_hand[i] = np.empty((0, 3))

        # Calculate contact points for RIGHT hand
        diff_right = right_hand_verts_world.unsqueeze(1) - obj_verts_sampled.unsqueeze(0)
        distances_right = torch.norm(diff_right, dim=-1)
        min_dist_per_obj_right, _ = distances_right.min(dim=0)
        obj_contact_mask_right = min_dist_per_obj_right < contact_threshold
        obj_contact_indices_sampled_right = torch.where(obj_contact_mask_right)[0]
        obj_contact_indices_right = downsample_indices[obj_contact_indices_sampled_right]

        if len(obj_contact_indices_right) > 0:
            obj_contact_points_world_right = obj_verts_world[obj_contact_indices_right]
            obj_contact_points_local_right = (
                obj_R[i].T @ (obj_contact_points_world_right - obj_t[i]).T
            ).T
            obj_contact_points_right_hand[i] = obj_contact_points_local_right.numpy()
            if right_hand_inter_start_idx == -1:
                right_hand_inter_start_idx = i
        else:
            obj_contact_points_right_hand[i] = np.empty((0, 3))

    obj_contact_points = {
        "left_hand": obj_contact_points_left_hand,
        "right_hand": obj_contact_points_right_hand,
    }
    obj_data["obj_contact_points"] = obj_contact_points
    print(
        f"Left hand inter start idx: {left_hand_inter_start_idx}, right hand inter start idx: {right_hand_inter_start_idx}, inter start idx: {inter_start_idx}"
    )

    # table_center_global = torch.tensor([8.25, 9, 0.0]) #p3
    # table_center_global = torch.tensor([5.7, 13.75, 0.0]) #p4
    table_center_global = None
    if hoi_data.get("scene_data", None) is not None and "table1" in hoi_data["scene_data"]:
        table_center_global = torch.tensor(hoi_data["scene_data"]["table1"]["pos"])

    if center_at_object or table_center_global is None:
        human_offset = obj_t[0, :3].clone()
    else:
        human_offset = table_center_global
    # human stands on the ground in the initial frame
    human_offset[2] = verts[0, :, 2].min()

    obj_offset = human_offset.clone()
    obj_vertices_world_first = (obj_R[0] @ torch.from_numpy(obj_vertices).float().T).T + obj_t[0]
    min_obj_z_first = obj_vertices_world_first[:, 2].min() - human_offset[2]
    obj_vertices_world_last = (obj_R[-1] @ torch.from_numpy(obj_vertices).float().T).T + obj_t[-1]
    min_obj_z_last = obj_vertices_world_last[:, 2].min() - human_offset[2]

    # HACK for on the ground object initially below the character
    if min_obj_z_first < 0.0:
        obj_offset[2] += min_obj_z_first

    # ========== Apply offsets to human_data and obj_data ==========
    human_data["trans"] = (human_trans - human_offset).numpy()
    if "joints" in human_data:
        human_data["joints"] = human_data["joints"] - human_offset.numpy()
    obj_data["obj_t"] = (obj_t - obj_offset).numpy()

    # Create fake table data with box
    table_depth = 0.04
    original_scene_data = hoi_data.get("scene_data", None)
    if (
        original_scene_data is not None
        and "table1" in original_scene_data
        and obj_data["obj_t"][0, 1] > 0.5
    ):
        # object is placed on the table later, so table height is the ending position of the object
        # the condition is temporary, to be improved later
        table_surface_height = obj_vertices_world_last[:, 2].min() - obj_offset[2]

        hoi_data["scene_data"] = {
            "table": {
                "pos": np.array((0, 0, table_surface_height.item() - table_depth / 2)),
                "size": np.array((1.0, 0.5, table_depth)),
            }
        }
        print(f"Table surface height: {hoi_data['scene_data']['table']['pos'][2]:.6f}")
    elif table_center_global is not None:
        # object is initially on the table, so table height is the starting position of the object
        table_surface_height = obj_vertices_world_first[:, 2].min() - obj_offset[2]
        table_length = 2.0
        table_width = 1.0
        if hoi_data.get("scene_data", None) is not None and "table1" in hoi_data["scene_data"]:
            table_length *= hoi_data["scene_data"]["table1"]["scale"][0]
            table_width *= hoi_data["scene_data"]["table1"]["scale"][2]

        hoi_data["scene_data"] = {
            "table": {
                "pos": np.array((0, 0, table_surface_height.item() - table_depth / 2)),
                "size": np.array((table_length, table_width, table_depth)),
            }
        }
        print(f"Saved table size: ({table_length:.2f}, {table_width:.2f}, {table_depth:.2f})")
        print(
            "Saved table surface height: {:.6f}".format(hoi_data["scene_data"]["table"]["pos"][2])
        )
    else:
        hoi_data["scene_data"] = None

    print(
        f"Applied offsets - human_offset: {human_offset.numpy()}, obj_offset: {obj_offset.numpy()}"
    )

    # save the scaled obj mesh
    obj_mesh.apply_scale(obj_scale)
    obj_mesh.export(f"{mesh_data_dir}/model.obj")

    # clean up meta
    hoi_data["meta"].pop("obj_path")
    hoi_data["meta"].pop("obj_pose_file")
    hoi_data["meta"].pop("render_config_file")
    hoi_data["meta"].pop("masks_cache_file")

    return hoi_data
