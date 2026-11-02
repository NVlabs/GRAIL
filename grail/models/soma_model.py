"""
SOMA model utilities for the HOI optimizer.

This module provides functions equivalent to those in smplx_model.py but for the SOMA body model.
"""

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

# SOMA 77-joint skeleton names (from nvskel77_name)
SOMA_JOINT_NAMES = [
    "Hips",  # 0
    "Spine1",  # 1
    "Spine2",  # 2
    "Chest",  # 3
    "Neck1",  # 4
    "Neck2",  # 5
    "Head",  # 6
    "HeadEnd",  # 7
    "Jaw",  # 8
    "LeftEye",  # 9
    "RightEye",  # 10
    # Left arm
    "LeftShoulder",  # 11
    "LeftArm",  # 12
    "LeftForeArm",  # 13
    "LeftHand",  # 14
    "LeftHandThumb1",  # 15
    "LeftHandThumb2",  # 16
    "LeftHandThumb3",  # 17
    "LeftHandThumbEnd",  # 18
    "LeftHandIndex1",  # 19
    "LeftHandIndex2",  # 20
    "LeftHandIndex3",  # 21
    "LeftHandIndex4",  # 22
    "LeftHandIndexEnd",  # 23
    "LeftHandMiddle1",  # 24
    "LeftHandMiddle2",  # 25
    "LeftHandMiddle3",  # 26
    "LeftHandMiddle4",  # 27
    "LeftHandMiddleEnd",  # 28
    "LeftHandRing1",  # 29
    "LeftHandRing2",  # 30
    "LeftHandRing3",  # 31
    "LeftHandRing4",  # 32
    "LeftHandRingEnd",  # 33
    "LeftHandPinky1",  # 34
    "LeftHandPinky2",  # 35
    "LeftHandPinky3",  # 36
    "LeftHandPinky4",  # 37
    "LeftHandPinkyEnd",  # 38
    # Right arm
    "RightShoulder",  # 39
    "RightArm",  # 40
    "RightForeArm",  # 41
    "RightHand",  # 42
    "RightHandThumb1",  # 43
    "RightHandThumb2",  # 44
    "RightHandThumb3",  # 45
    "RightHandThumbEnd",  # 46
    "RightHandIndex1",  # 47
    "RightHandIndex2",  # 48
    "RightHandIndex3",  # 49
    "RightHandIndex4",  # 50
    "RightHandIndexEnd",  # 51
    "RightHandMiddle1",  # 52
    "RightHandMiddle2",  # 53
    "RightHandMiddle3",  # 54
    "RightHandMiddle4",  # 55
    "RightHandMiddleEnd",  # 56
    "RightHandRing1",  # 57
    "RightHandRing2",  # 58
    "RightHandRing3",  # 59
    "RightHandRing4",  # 60
    "RightHandRingEnd",  # 61
    "RightHandPinky1",  # 62
    "RightHandPinky2",  # 63
    "RightHandPinky3",  # 64
    "RightHandPinky4",  # 65
    "RightHandPinkyEnd",  # 66
    # Left leg
    "LeftLeg",  # 67
    "LeftShin",  # 68
    "LeftFoot",  # 69
    "LeftToeBase",  # 70
    "LeftToeEnd",  # 71
    # Right leg
    "RightLeg",  # 72
    "RightShin",  # 73
    "RightFoot",  # 74
    "RightToeBase",  # 75
    "RightToeEnd",  # 76
]

# Mapping from body segment names to SOMA joint indices for vertex segmentation
# The joint indices correspond to the root joint of each body part
SOMA_SEGMENT_JOINT_INDICES = {
    "L_Hand": 14,  # LeftHand
    "R_Hand": 42,  # RightHand
    "L_Foot": 69,  # LeftFoot
    "R_Foot": 74,  # RightFoot
    "L_Wrist": 14,  # LeftHand (same as L_Hand for SOMA)
    "R_Wrist": 42,  # RightHand (same as R_Hand for SOMA)
    "L_Ankle": 69,  # LeftFoot
    "R_Ankle": 74,  # RightFoot
    "Head": 6,  # Head
    "L_Shoulder": 12,  # LeftArm
    "R_Shoulder": 40,  # RightArm
    "L_Elbow": 13,  # LeftForeArm
    "R_Elbow": 41,  # RightForeArm
    "L_Knee": 68,  # LeftShin
    "R_Knee": 73,  # RightShin
    "L_Hip": 67,  # LeftLeg
    "R_Hip": 72,  # RightLeg
}

# ============================================================================
# SOMA Joint Index Constants for Optimization
# ============================================================================
# These constants define which joints belong to body vs hands for pose residual optimization.
# The body_pose tensor has 76 joints (indices 0-75), excluding root which is global_orient.
# The full skeleton has 77 joints (indices 0-76), including root at index 0.

# Body joints in body_pose tensor (28 joints: spine, head, arms including wrists, legs)
# Excludes finger joints which are handled separately for hand optimization
# Body joints in body_pose tensor (26 joints: spine, head, arms excluding wrists, legs)
# Wrists are included in hand joints for consistency with SMPL-X implementation
SOMA_BODY_POSE_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,  # Spine to left forearm (13 joints)
    38,
    39,
    40,  # Right arm: shoulder, arm, forearm (3 joints)
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,  # Legs (10 joints)
]  # 26 body joints in body_pose tensor

# Left hand joints in body_pose (25 joints: wrist at 13 + fingers 14-37)
SOMA_LEFT_HAND_POSE_INDICES = list(range(13, 38))

# Right hand joints in body_pose (25 joints: wrist at 41 + fingers 42-65)
SOMA_RIGHT_HAND_POSE_INDICES = list(range(41, 66))

# Full skeleton joint indices (77 joints including root at index 0)
# Body joints: root + body_pose body joints = 27 joints
SOMA_BODY_JOINT_INDICES = [0] + [i + 1 for i in SOMA_BODY_POSE_INDICES]  # 27 joints

# Hand joints in full skeleton (wrist + 24 finger joints per hand = 25 per hand, 50 total)
SOMA_LEFT_HAND_JOINT_INDICES = list(range(14, 39))  # Left wrist (14) + fingers (15-38)
SOMA_RIGHT_HAND_JOINT_INDICES = list(range(42, 67))  # Right wrist (42) + fingers (43-66)


# Cache for vertex segment indices
_SOMA_SEGMENT_VERTEX_CACHE = {}


def setup_soma_model(model_path=None, device="cuda"):
    """Refuse the retired hybrid MHR runtime rather than reinterpret its motion."""
    raise ValueError(
        "The legacy MHR-backed soma runtime is retired. Use the archived code for "
        "historical checkpoints; new reconstruction uses soma03 with GEM-X/SOMA-X. "
        "Changing only the model tag is not a valid checkpoint conversion."
    )


def load_soma_beta(beta_path, device="cuda"):
    """Load SOMA shape parameters (identity_coeffs and scale_params) from a pickle file.

    Args:
        beta_path: Path to the .pkl file containing SOMA shape parameters.
                   Expected to contain 'identity_coeffs' (45,) and 'scale_params' (75,).
        device: Device to load tensors to.

    Returns:
        tuple: (identity_coeffs, scale_params) as torch tensors with shape (1, 45) and (1, 75).
    """
    import pickle

    with open(beta_path, "rb") as f:
        data = pickle.load(f)

    identity_coeffs = torch.tensor(data["identity_coeffs"], dtype=torch.float32, device=device)
    scale_params = torch.tensor(data["scale_params"], dtype=torch.float32, device=device)

    # Add batch dimension if needed
    if identity_coeffs.dim() == 1:
        identity_coeffs = identity_coeffs.unsqueeze(0)  # (1, 45)
    if scale_params.dim() == 1:
        scale_params = scale_params.unsqueeze(0)  # (1, 75)

    return identity_coeffs, scale_params


def forward_soma(soma_model, motion_data, require_grad=False, device="cuda"):
    """
    Forward pass through the SOMA model.

    Args:
        soma_model: NovaLayer instance
        motion_data: Dictionary containing SOMA parameters. Supports multiple formats:

            Key naming (both supported):
                - SOMA native: "body_pose", "transl", "global_orient"
                - SMPL-X style: "poses", "trans"

            Body pose formats:
                Format 1 (separate global_orient):
                    - global_orient: (L, 3) global orientation in axis-angle
                    - body_pose/poses: (L, 76, 3) or (L, 228) body pose in axis-angle
                Format 2 (combined):
                    - body_pose/poses: (L, 77, 3) or (L, 231) where first joint is global_orient

            Required parameters:
                - identity_coeffs: (45,) or (L, 45) identity shape coefficients
                - scale_params: (75,) or (L, 75) scale parameters
                - transl/trans: (L, 3) translation

        require_grad: Whether to enable gradients
        device: Device for computation

    Returns:
        dict: SOMA output with "vertices" (L, V, 3) and "joints" (L, 77, 3)
    """
    # Handle different key naming conventions:
    # - SOMA native: body_pose, transl, global_orient
    # - SMPL-X style: poses, trans (combined format)

    # Get body pose (supports both "body_pose" and "poses" keys)
    if "body_pose" in motion_data:
        body_pose = motion_data["body_pose"]
    elif "poses" in motion_data:
        body_pose = motion_data["poses"]
    else:
        raise KeyError("motion_data must contain either 'body_pose' or 'poses' key")

    # Get identity coeffs and scale params
    identity_coeffs = motion_data["identity_coeffs"]
    scale_params = motion_data["scale_params"]

    # Get translation (supports both "transl" and "trans" keys)
    if "transl" in motion_data:
        transl = motion_data["transl"]
    elif "trans" in motion_data:
        transl = motion_data["trans"]
    else:
        raise KeyError("motion_data must contain either 'transl' or 'trans' key")

    # Convert body_pose to tensor first to check its shape
    if not isinstance(body_pose, torch.Tensor):
        body_pose = torch.tensor(body_pose, device=device, dtype=torch.float32)
    body_pose = body_pose.to(device)

    # Check if global_orient is provided separately or needs to be extracted from body_pose
    if "global_orient" in motion_data:
        global_orient = motion_data["global_orient"]
        if not isinstance(global_orient, torch.Tensor):
            global_orient = torch.tensor(global_orient, device=device, dtype=torch.float32)
        global_orient = global_orient.to(device)
        frame_num = global_orient.shape[0]

        # Reshape body_pose if it's flattened (76 joints * 3 = 228)
        if body_pose.dim() == 2 and body_pose.shape[-1] == 228:
            body_pose = body_pose.reshape(frame_num, 76, 3)
    else:
        # Extract global_orient from body_pose (first joint)
        # body_pose can be (L, 77, 3) or (L, 231)
        if body_pose.dim() == 2:
            # Flattened format
            if body_pose.shape[-1] == 231:
                # 77 joints * 3 = 231
                frame_num = body_pose.shape[0]
                body_pose = body_pose.reshape(frame_num, 77, 3)
            elif body_pose.shape[-1] == 228:
                # 76 joints * 3 = 228, no global_orient included - this shouldn't happen
                raise ValueError("body_pose has 76 joints but global_orient is not provided")
            else:
                raise ValueError(f"Unexpected body_pose shape: {body_pose.shape}")

        frame_num = body_pose.shape[0]

        if body_pose.shape[1] == 77:
            # Extract global_orient from first joint
            global_orient = body_pose[:, 0, :]  # (L, 3)
            body_pose = body_pose[:, 1:, :]  # (L, 76, 3)
        elif body_pose.shape[1] == 76:
            raise ValueError("body_pose has 76 joints but global_orient is not provided")
        else:
            raise ValueError(f"Unexpected body_pose joint count: {body_pose.shape[1]}")

    # Convert remaining parameters to tensors if necessary
    if not isinstance(identity_coeffs, torch.Tensor):
        identity_coeffs = torch.tensor(identity_coeffs, device=device, dtype=torch.float32)
    if not isinstance(scale_params, torch.Tensor):
        scale_params = torch.tensor(scale_params, device=device, dtype=torch.float32)
    if not isinstance(transl, torch.Tensor):
        transl = torch.tensor(transl, device=device, dtype=torch.float32)

    # Move to device
    identity_coeffs = identity_coeffs.to(device)
    scale_params = scale_params.to(device)
    transl = transl.to(device)

    # Reshape body_pose if it's still flattened
    if body_pose.dim() == 2 and body_pose.shape[-1] == 228:
        body_pose = body_pose.reshape(frame_num, 76, 3)

    # Handle identity coeffs and scale params - they can be shared across frames
    # Need to repeat to match frame count (L, 45) and (L, 75)
    if identity_coeffs.dim() == 1:
        identity_coeffs = identity_coeffs.unsqueeze(0).repeat(frame_num, 1)  # (L, 45)
    elif identity_coeffs.shape[0] == 1 and frame_num > 1:
        identity_coeffs = identity_coeffs.repeat(frame_num, 1)  # (L, 45)
    if scale_params.dim() == 1:
        scale_params = scale_params.unsqueeze(0).repeat(frame_num, 1)  # (L, 75)
    elif scale_params.shape[0] == 1 and frame_num > 1:
        scale_params = scale_params.repeat(frame_num, 1)  # (L, 75)

    # SomaLayer wrapper expects global_orient + body_pose with batch dim
    # Add batch dimension (B=1)
    global_orient = global_orient.unsqueeze(0)  # (1, L, 3)
    body_pose = body_pose.unsqueeze(0)  # (1, L, 76, 3)
    identity_coeffs = identity_coeffs.unsqueeze(0)  # (1, L, 45)
    scale_params = scale_params.unsqueeze(0)  # (1, L, 69)
    transl = transl.unsqueeze(0)  # (1, L, 3)

    if require_grad:
        output = soma_model(
            global_orient=global_orient,
            body_pose=body_pose,
            identity_coeffs=identity_coeffs,
            scale_params=scale_params,
            transl=transl,
        )
    else:
        with torch.no_grad():
            output = soma_model(
                global_orient=global_orient,
                body_pose=body_pose,
                identity_coeffs=identity_coeffs,
                scale_params=scale_params,
                transl=transl,
            )

    # Remove batch dimension from output
    result = {
        "vertices": output["vertices"].squeeze(0),  # (L, V, 3)
        "joints": output["joints"].squeeze(0),  # (L, 77, 3)
    }

    return result


def generate_soma_mesh(
    soma_model, motion_data, output_joints=False, require_grad=False, device="cuda"
):
    """
    Generate SOMA mesh vertices and faces from motion data.

    Args:
        soma_model: NovaLayer instance
        motion_data: Dictionary containing SOMA parameters
        output_joints: Whether to also return joint positions
        require_grad: Whether to enable gradients
        device: Device for computation

    Returns:
        If output_joints=False:
            (vertices, faces): Tuple of vertices (L, V, 3) and faces tensor
        If output_joints=True:
            (vertices, faces, joints): Including joint positions (L, 77, 3)
    """
    output = forward_soma(soma_model, motion_data, require_grad, device)
    vertices = output["vertices"]
    joints = output["joints"]

    # Get faces from the model
    faces = soma_model.faces
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).to(device).long()
    elif isinstance(faces, torch.Tensor):
        faces = faces.to(device).long()

    # Apply scale if present in motion_data
    scale = motion_data.get("scale", 1.0)
    if scale != 1.0:
        # Support both 'transl' (NOVA) and 'trans' (SMPL-X) key names
        if "transl" in motion_data:
            transl = motion_data["transl"]
        elif "trans" in motion_data:
            transl = motion_data["trans"]
        else:
            raise KeyError("motion_data must contain either 'transl' or 'trans' key for scaling")

        if not isinstance(transl, torch.Tensor):
            transl = torch.tensor(transl, device=device, dtype=torch.float32)
        transl = transl.to(device)

        frame_num = vertices.shape[0]
        transl = transl.reshape(frame_num, 1, 3)

        # Scale vertices around translation point
        vertices = (vertices - transl) * scale + transl

        # Scale joints as well
        joints = (joints - transl) * scale + transl

    if output_joints:
        return vertices, faces, joints
    else:
        return vertices, faces


def get_soma_verts_segment(soma_verts, segment_name=["L_Hand", "R_Hand"], soma_model=None):
    """
    Get vertices belonging to specific body segments.

    Uses skinning weights from the SOMA model to determine which vertices belong
    to each body segment. The segment is defined by a root joint and includes
    all vertices that are influenced by that joint or its descendants.

    Args:
        soma_verts: Vertex positions (L, V, 3) or (V, 3)
        segment_name: List of segment names (e.g., ["L_Hand", "R_Hand"])
                     Available segments: L_Hand, R_Hand, L_Foot, R_Foot,
                     L_Wrist, R_Wrist, L_Ankle, R_Ankle, Head,
                     L_Shoulder, R_Shoulder, L_Elbow, R_Elbow,
                     L_Knee, R_Knee, L_Hip, R_Hip
        soma_model: NovaLayer instance (required for computing segment indices
                    on first call; cached thereafter)

    Returns:
        torch.Tensor: Vertices for the specified segments (L, N, 3) or (N, 3)
    """
    global _SOMA_SEGMENT_VERTEX_CACHE

    if soma_verts.dim() == 2:
        soma_verts = soma_verts.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if soma_verts.dim() != 3:
        raise ValueError("soma_verts must be a 2D or 3D tensor")

    device = soma_verts.device
    segment_verts_list = []

    for name in segment_name:
        if name not in SOMA_SEGMENT_JOINT_INDICES:
            raise ValueError(
                f"Unknown segment name: {name}. Available: {list(SOMA_SEGMENT_JOINT_INDICES.keys())}"
            )

        # Check cache first - use segment name and vertex count as cache key
        # (vertex count distinguishes between different LOD models)
        num_verts = soma_verts.shape[1]
        cache_key = (name, num_verts)

        if cache_key in _SOMA_SEGMENT_VERTEX_CACHE:
            vertex_ids = _SOMA_SEGMENT_VERTEX_CACHE[cache_key]
        else:
            if soma_model is None:
                raise ValueError("soma_model is required to compute segment vertex indices")

            from soma.geometry.rig_utils import get_body_part_vertex_ids

            joint_id = SOMA_SEGMENT_JOINT_INDICES[name]

            try:
                skinning_weights = getattr(soma_model, "skinning_weights", None)
                if skinning_weights is None:
                    skinning_weights = getattr(soma_model, "soma", soma_model).skinning_weights
                if skinning_weights.device.type != "cpu":
                    skinning_weights = skinning_weights.cpu()

                # joint_parent_ids is 0-indexed in the 78-joint space
                # (joint 0 = "World" auxiliary, joints 1-77 = actual skeleton).
                # skinning_weights has shape (V, 78) in the same space.
                # SOMA_SEGMENT_JOINT_INDICES is in 77-joint naming, so +1
                # to convert to the 78-joint space that parent IDs and
                # skinning weight columns use.
                raw_parent_ids = getattr(soma_model, "joint_parent_ids", None)
                if raw_parent_ids is None:
                    raw_parent_ids = getattr(soma_model, "soma", soma_model).parents
                if isinstance(raw_parent_ids, torch.Tensor):
                    raw_parent_ids = raw_parent_ids.cpu().tolist()
                joint_id_78 = joint_id + 1

                vertex_ids = get_body_part_vertex_ids(
                    skinning_weights,
                    raw_parent_ids,
                    joint_id_78,
                    include_root=True,
                    weight_threshold=0.1,
                )
            except (AttributeError, KeyError) as e:
                raise RuntimeError(
                    f"Cannot access skinning weights from SOMA model: {e}. "
                    "Please ensure the model has 'skinning_weights' attribute."
                )

            _SOMA_SEGMENT_VERTEX_CACHE[cache_key] = vertex_ids

        # Gather vertices for this segment
        # Convert vertex_ids to tensor if it's a list
        if isinstance(vertex_ids, list):
            vertex_ids_tensor = torch.tensor(vertex_ids, device=device, dtype=torch.long)
        else:
            vertex_ids_tensor = vertex_ids.to(device)

        segment_verts = soma_verts[:, vertex_ids_tensor, :]
        segment_verts_list.append(segment_verts)

    # Concatenate all segments
    result = torch.cat(segment_verts_list, dim=1)

    if squeeze_output:
        result = result.squeeze(0)

    return result


def get_soma_segment_indices(segment_name=["L_Hand", "R_Hand"], soma_model=None):
    """
    Return vertex indices for specified body segments.

    Similar to get_smplx_segment_indices but for SOMA model.
    Uses skinning weights from the SOMA model to determine which vertices belong
    to each body segment.

    Args:
        segment_name: List of segment names (e.g., ["L_Hand", "R_Hand"])
                     Available segments: L_Hand, R_Hand, L_Foot, R_Foot,
                     L_Wrist, R_Wrist, L_Ankle, R_Ankle, Head,
                     L_Shoulder, R_Shoulder, L_Elbow, R_Elbow,
                     L_Knee, R_Knee, L_Hip, R_Hip
        soma_model: NovaLayer instance (required for computing segment indices
                    on first call; cached thereafter)

    Returns:
        List of vertex indices for the specified segments
    """
    global _SOMA_SEGMENT_VERTEX_CACHE

    indices = []

    for name in segment_name:
        if name not in SOMA_SEGMENT_JOINT_INDICES:
            raise ValueError(
                f"Unknown segment name: {name}. Available: {list(SOMA_SEGMENT_JOINT_INDICES.keys())}"
            )

        # Use a default vertex count for caching (assuming standard SOMA model)
        # The actual vertex count may vary, but for indices-only function we use a placeholder
        num_verts = "default"
        cache_key = (name, num_verts)

        if cache_key in _SOMA_SEGMENT_VERTEX_CACHE:
            vertex_ids = _SOMA_SEGMENT_VERTEX_CACHE[cache_key]
        else:
            if soma_model is None:
                raise ValueError("soma_model is required to compute segment vertex indices")

            from soma.geometry.rig_utils import get_body_part_vertex_ids

            joint_id = SOMA_SEGMENT_JOINT_INDICES[name]

            try:
                skinning_weights = getattr(soma_model, "skinning_weights", None)
                if skinning_weights is None:
                    skinning_weights = getattr(soma_model, "soma", soma_model).skinning_weights
                if skinning_weights.device.type != "cpu":
                    skinning_weights = skinning_weights.cpu()

                raw_parent_ids = getattr(soma_model, "joint_parent_ids", None)
                if raw_parent_ids is None:
                    raw_parent_ids = getattr(soma_model, "soma", soma_model).parents
                if isinstance(raw_parent_ids, torch.Tensor):
                    raw_parent_ids = raw_parent_ids.cpu().tolist()
                joint_id_78 = joint_id + 1

                vertex_ids = get_body_part_vertex_ids(
                    skinning_weights,
                    raw_parent_ids,
                    joint_id_78,
                    include_root=True,
                    weight_threshold=0.1,
                )
            except (AttributeError, KeyError) as e:
                raise RuntimeError(
                    f"Cannot access skinning weights from SOMA model: {e}. "
                    "Please ensure the model has 'skinning_weights' attribute."
                )

            _SOMA_SEGMENT_VERTEX_CACHE[cache_key] = vertex_ids

        # Extend indices list
        if isinstance(vertex_ids, list):
            indices.extend(vertex_ids)
        else:
            indices.extend(vertex_ids.tolist())

    return indices


def get_tpose_human_height(soma_model, identity_coeffs=None, scale_params=None, device="cuda"):
    """
    Generate T-pose human and calculate height.

    Args:
        soma_model: NovaLayer instance
        identity_coeffs: Shape coefficients (45,) tensor. If None, uses neutral shape
        scale_params: Scale parameters (75,) tensor. If None, uses neutral scale
        device: Device for computation

    Returns:
        float: Height of T-pose human in meters
    """
    # Create default identity coefficients if not provided
    if identity_coeffs is None:
        identity_coeffs = torch.zeros(45, device=device, dtype=torch.float32)
    elif not isinstance(identity_coeffs, torch.Tensor):
        identity_coeffs = torch.tensor(identity_coeffs, device=device, dtype=torch.float32)

    if identity_coeffs.dim() == 2:
        identity_coeffs = identity_coeffs[0]  # Take first frame if batched

    # Create default scale params if not provided
    if scale_params is None:
        scale_params = torch.zeros(75, device=device, dtype=torch.float32)
    elif not isinstance(scale_params, torch.Tensor):
        scale_params = torch.tensor(scale_params, device=device, dtype=torch.float32)

    if scale_params.dim() == 2:
        scale_params = scale_params[0]  # Take first frame if batched

    # Ensure identity_coeffs and scale_params are 2D (L, D) for consistency
    if identity_coeffs.dim() == 1:
        identity_coeffs = identity_coeffs.unsqueeze(0)  # (1, 45)
    if scale_params.dim() == 1:
        scale_params = scale_params.unsqueeze(0)  # (1, 69)

    # Create T-pose motion data (all zeros = T-pose)
    tpose_data = {
        "global_orient": torch.zeros(1, 3, device=device, dtype=torch.float32),
        "body_pose": torch.zeros(1, 76, 3, device=device, dtype=torch.float32),
        "identity_coeffs": identity_coeffs,
        "scale_params": scale_params,
        "transl": torch.zeros(1, 3, device=device, dtype=torch.float32),
    }

    # Generate T-pose mesh
    tpose_verts, _ = generate_soma_mesh(
        soma_model,
        tpose_data,
        output_joints=False,
        require_grad=False,
        device=device,
    )

    tpose_human_verts = tpose_verts[0]  # (V, 3)

    # Calculate height (max Y - min Y)
    # SOMA Y-axis is the vertical axis
    height = tpose_human_verts[:, 1].max() - tpose_human_verts[:, 1].min()

    return height


def transform_global_to_incam(global_motion_data, incam_motion_data, align_frame=0, device="cuda"):
    """
    Transform global motion data to in-camera coordinate frame.

    This function computes a transformation matrix that aligns the global motion data
    with the in-camera motion data at a specific reference frame, then applies this
    transformation to all frames in the global motion data.

    Args:
        global_motion_data: Global motion data containing poses and translation.
            Supports both SOMA style ('global_orient', 'transl') and SMPL-X style ('poses', 'trans').
        incam_motion_data: In-camera motion data with same key conventions.
        align_frame: Reference frame for alignment (default: 0)
        device: Device for computations

    Returns:
        dict: Transformed global motion data in camera coordinate frame
    """
    # Make a copy to avoid modifying the original data
    transformed_motion_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_motion_data.items()
    }

    # Copy over any keys from incam_motion_data that are not in global_motion_data
    for k, v in incam_motion_data.items():
        if k not in transformed_motion_data.keys():
            transformed_motion_data[k] = v.clone() if isinstance(v, torch.Tensor) else v

    # Determine key names (SMPL-X style: trans/poses vs SOMA style: transl/global_orient)
    trans_key = "trans" if "trans" in global_motion_data else "transl"

    # Check if global_orient is separate or combined with poses
    if "global_orient" in global_motion_data:
        has_separate_global_orient = True
    else:
        has_separate_global_orient = False
        poses_key = "poses" if "poses" in global_motion_data else "body_pose"

    # Helper function to get global orientation
    def get_global_orient(motion_data, frame_idx):
        if has_separate_global_orient:
            return motion_data["global_orient"][frame_idx]
        else:
            return motion_data[poses_key][frame_idx, :3]

    # Extract reference orientations and translations at the alignment frame
    init_global_t = global_motion_data[trans_key][align_frame]
    init_incam_t = incam_motion_data[trans_key][align_frame]
    init_global_R = axis_angle_to_matrix(get_global_orient(global_motion_data, align_frame))
    init_incam_R = axis_angle_to_matrix(get_global_orient(incam_motion_data, align_frame))

    # Compute 4x4 transformation matrices for global and in-camera poses
    init_global_pose = torch.eye(4, device=device, dtype=torch.float32)
    init_global_pose[:3, :3] = init_global_R
    init_global_pose[:3, 3] = init_global_t

    init_incam_pose = torch.eye(4, device=device, dtype=torch.float32)
    init_incam_pose[:3, :3] = init_incam_R
    init_incam_pose[:3, 3] = init_incam_t

    # Compute transformation from global to in-camera coordinate frame
    global_to_incam_pose = init_incam_pose @ torch.linalg.inv(init_global_pose)

    # Transform all frames from global to in-camera coordinate frame
    trans_len = global_motion_data[trans_key].shape[0]
    for i in range(trans_len):
        # Convert current frame to 4x4 transformation matrix
        global_t = global_motion_data[trans_key][i]
        global_R = axis_angle_to_matrix(get_global_orient(global_motion_data, i))
        global_pose = torch.eye(4, device=device, dtype=torch.float32)
        global_pose[:3, :3] = global_R
        global_pose[:3, 3] = global_t

        # Apply transformation to get in-camera pose
        incam_pose = global_to_incam_pose @ global_pose

        # Extract transformed translation and rotation
        transformed_motion_data[trans_key][i] = incam_pose[:3, 3].reshape(3)
        new_global_orient = matrix_to_axis_angle(incam_pose[:3, :3])
        if has_separate_global_orient:
            transformed_motion_data["global_orient"][i] = new_global_orient
        else:
            transformed_motion_data[poses_key][i, :3] = new_global_orient

    return transformed_motion_data


def transform_global_motion(
    soma_model,
    global_motion_data,
    incam_motion_data,
    cam_R=None,
    cam_t=None,
    align_frame=0,
    use_global=False,
    gt_identity_coeffs=None,
    gt_scale_params=None,
    device="cuda",
):
    """
    Transform global motion to camera frame with optional human scaling.

    Args:
        soma_model: NovaLayer instance
        global_motion_data: Global motion data
        incam_motion_data: In-camera motion data
        cam_R: Camera rotation matrix (3x3)
        cam_t: Camera translation (3,)
        align_frame: Reference frame for alignment
        use_global: Whether to use global motion data
        gt_identity_coeffs: Ground truth identity coefficients for scaling
        gt_scale_params: Ground truth scale parameters for scaling
        device: Device for computation

    Returns:
        dict: Transformed motion data
    """
    if cam_R is None:
        cam_R = torch.eye(3, device=device, dtype=torch.float32)
    if cam_t is None:
        cam_t = torch.zeros(3, device=device, dtype=torch.float32)

    if use_global:
        transformed_incam_motion_data = transform_global_to_incam(
            global_motion_data, incam_motion_data, align_frame=align_frame, device=device
        )
    else:
        transformed_incam_motion_data = incam_motion_data

    transformed_motion_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in transformed_incam_motion_data.items()
    }

    # Determine key names (SMPL-X style: trans/poses vs SOMA style: transl/global_orient/body_pose)
    trans_key = "trans" if "trans" in transformed_incam_motion_data else "transl"

    # Check if global_orient is separate or combined with poses
    if "global_orient" in transformed_incam_motion_data:
        has_separate_global_orient = True
    else:
        has_separate_global_orient = False
        # global_orient is the first 3 values of poses
        poses_key = "poses" if "poses" in transformed_incam_motion_data else "body_pose"

    # Get the root joint positions
    _, _, incam_joints = generate_soma_mesh(
        soma_model,
        transformed_incam_motion_data,
        output_joints=True,
        require_grad=False,
        device=device,
    )

    incam_root_joint_seq = incam_joints[:, 0, :]  # Root joint (Hips)

    frame_num = transformed_incam_motion_data[trans_key].shape[0]
    for i in range(frame_num):
        incam_t = transformed_incam_motion_data[trans_key][i]

        # Get global orientation
        if has_separate_global_orient:
            incam_global_orient = transformed_incam_motion_data["global_orient"][i]
        else:
            # Extract from poses (first 3 values)
            incam_global_orient = transformed_incam_motion_data[poses_key][i, :3]

        incam_R = axis_angle_to_matrix(incam_global_orient)
        diff = incam_root_joint_seq[i] - incam_t
        transformed_motion_data[trans_key][i] = cam_R @ incam_t + cam_t + cam_R @ diff - diff

        # Update global orientation
        new_global_orient = matrix_to_axis_angle(cam_R @ incam_R)
        if has_separate_global_orient:
            transformed_motion_data["global_orient"][i] = new_global_orient
        else:
            # Update the first 3 values of poses
            transformed_motion_data[poses_key][i, :3] = new_global_orient

    # Apply human scaling if ground truth identity is provided
    if gt_identity_coeffs is not None and gt_scale_params is not None:
        est_tpose_height = get_tpose_human_height(
            soma_model,
            transformed_incam_motion_data["identity_coeffs"],
            transformed_incam_motion_data["scale_params"],
            device=device,
        )
        gt_tpose_height = get_tpose_human_height(
            soma_model, gt_identity_coeffs, gt_scale_params, device=device
        )
        human_scale_ratio = gt_tpose_height / est_tpose_height

        transformed_motion_data[trans_key] = (
            transformed_motion_data[trans_key] - cam_t
        ) * human_scale_ratio + cam_t
        transformed_motion_data["identity_coeffs"] = gt_identity_coeffs
        transformed_motion_data["scale_params"] = gt_scale_params
        transformed_motion_data["scale"] = 1.0
    else:
        transformed_motion_data["scale"] = 1.0

    return transformed_motion_data


def convert_smplx_to_soma_motion(
    smplx_motion_data, soma_identity_coeffs, soma_scale_params, device="cuda"
):
    """
    Convert SMPL-X motion data format to SOMA motion data format.

    Note: This is a simple conversion that maps the common parameters.
    Hand poses and other detailed mappings may require additional processing.

    Args:
        smplx_motion_data: SMPL-X motion data with 'poses', 'betas', 'trans'
        soma_identity_coeffs: SOMA identity coefficients (45,)
        soma_scale_params: SOMA scale parameters (75,)
        device: Device for computation

    Returns:
        dict: SOMA motion data format
    """
    poses = smplx_motion_data["poses"]
    trans = smplx_motion_data["trans"]

    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, device=device, dtype=torch.float32)
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans, device=device, dtype=torch.float32)

    frame_num = poses.shape[0]
    poses = poses.reshape(frame_num, -1)

    # Extract global orientation (first 3 values)
    global_orient = poses[:, :3]

    # SOMA body_pose has 76 joints x 3 = 228 values
    # SMPL-X body_pose has 21 joints x 3 = 63 values (after global orient)
    # We need to map SMPL-X joints to SOMA joints
    # For now, we just initialize SOMA body pose with zeros and copy what we can
    body_pose = torch.zeros(frame_num, 76, 3, device=device, dtype=torch.float32)

    # Copy body pose from SMPL-X (joints 1-21, which maps to specific SOMA joints)
    # This is a simplified mapping - actual joint correspondence would need more work
    smplx_body_pose = poses[:, 3 : 3 + 21 * 3].reshape(frame_num, 21, 3)

    # Rough mapping of SMPL-X body joints to SOMA body joints
    # SMPL-X: pelvis(0), left_hip(1), right_hip(2), spine1(3), left_knee(4), right_knee(5),
    #         spine2(6), left_ankle(7), right_ankle(8), spine3(9), left_foot(10), right_foot(11),
    #         neck(12), left_collar(13), right_collar(14), head(15), left_shoulder(16),
    #         right_shoulder(17), left_elbow(18), right_elbow(19), left_wrist(20), right_wrist(21)
    #
    # SOMA mapping (approximate):
    # 0: Hips, 1: Spine1, 2: Spine2, 3: Chest, 4-5: Neck, 6: Head
    # 11: LeftShoulder, 12: LeftArm, 13: LeftForeArm, 14: LeftHand
    # 39: RightShoulder, 40: RightArm, 41: RightForeArm, 42: RightHand
    # 67: LeftLeg, 68: LeftShin, 69: LeftFoot, 70: LeftToeBase
    # 72: RightLeg, 73: RightShin, 74: RightFoot, 75: RightToeBase

    # Simplified mapping (indices adjusted for 0-based indexing in body_pose which excludes root)
    smplx_to_soma = {
        0: 67,  # L_Hip -> LeftLeg (adjusted for body_pose indexing)
        1: 72,  # R_Hip -> RightLeg
        2: 1,  # Spine1
        3: 68,  # L_Knee -> LeftShin
        4: 73,  # R_Knee -> RightShin
        5: 2,  # Spine2
        6: 69,  # L_Ankle -> LeftFoot
        7: 74,  # R_Ankle -> RightFoot
        8: 3,  # Chest
        9: 70,  # L_Foot -> LeftToeBase
        10: 75,  # R_Foot -> RightToeBase
        11: 4,  # Neck -> Neck1
        12: 11,  # L_Collar -> LeftShoulder
        13: 39,  # R_Collar -> RightShoulder
        14: 6,  # Head
        15: 12,  # L_Shoulder -> LeftArm
        16: 40,  # R_Shoulder -> RightArm
        17: 13,  # L_Elbow -> LeftForeArm
        18: 41,  # R_Elbow -> RightForeArm
        19: 14,  # L_Wrist -> LeftHand
        20: 42,  # R_Wrist -> RightHand
    }

    for smplx_idx, soma_idx in smplx_to_soma.items():
        body_pose[:, soma_idx, :] = smplx_body_pose[:, smplx_idx, :]

    # Handle hand poses if present
    left_hand_pose = smplx_motion_data.get("left_hand_pose", None)
    right_hand_pose = smplx_motion_data.get("right_hand_pose", None)

    # Note: SOMA and SMPL-X have different hand joint orderings
    # This would require more careful mapping for accurate hand pose transfer

    soma_motion_data = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "identity_coeffs": soma_identity_coeffs,
        "scale_params": soma_scale_params,
        "transl": trans,
    }

    return soma_motion_data
