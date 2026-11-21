import torch
from pytorch3d.loss import point_mesh_distance


def l1_loss(A, B):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A.shape={A.shape}, B.shape={B.shape}")
    return torch.nn.functional.l1_loss(A, B)


def l2_loss(A, B):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A.shape={A.shape}, B.shape={B.shape}")
    return torch.nn.functional.mse_loss(A, B)


def huber_loss(a, delta=1e-2):
    abs_a = torch.abs(a)
    quadratic = 0.5 * abs_a**2
    linear = delta * (abs_a - 0.5 * delta)
    loss = torch.where(abs_a < delta, quadratic, linear)  # Huber Loss formula
    return loss.mean()


def bidirectional_chamfer_loss(pred_verts, gt_points, trim_pct=0.2):
    """
    Trimmed bidirectional Chamfer distance using knn_points for efficiency.

    pred->GT: each pred vertex finds nearest GT point (squared L2).
    GT->pred: each GT point finds nearest pred vertex (squared L2).
    Both directions are trimmed independently to handle GT depth noise.

    Args:
        pred_verts: (N, 3) predicted visible vertices (differentiable)
        gt_points: (M, 3) GT 3D point cloud from depth map (detached)
        trim_pct: fraction of worst matches to discard (0.0 = no trimming)

    Returns:
        torch.Tensor: scalar loss
    """
    from pytorch3d.ops import knn_points

    device = pred_verts.device
    if pred_verts.shape[0] == 0 or gt_points.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pred = pred_verts.float().unsqueeze(0)  # (1, N, 3)
    gt = gt_points.float().unsqueeze(0)  # (1, M, 3)

    # pred -> GT direction
    pred2gt_dists = knn_points(pred, gt, K=1).dists.squeeze(0).squeeze(-1)  # (N,)

    # GT -> pred direction
    gt2pred_dists = knn_points(gt, pred, K=1).dists.squeeze(0).squeeze(-1)  # (M,)

    # Trim outliers independently per direction
    k_p = max(1, int(len(pred2gt_dists) * (1.0 - trim_pct)))
    k_g = max(1, int(len(gt2pred_dists) * (1.0 - trim_pct)))
    trimmed_p, _ = pred2gt_dists.topk(k_p, largest=False)
    trimmed_g, _ = gt2pred_dists.topk(k_g, largest=False)

    return trimmed_p.mean() + trimmed_g.mean()


def contact_loss(verts_A, verts_B, num_vertices=2000, top_k=200, delta=0.001):
    """
    Calculate contact loss based on minimum distance between two point clouds.

    Args:
        verts_A (torch.Tensor): Point cloud A (N, 3)
        verts_B (torch.Tensor): Point cloud B (M, 3)
        num_vertices (int): Number of closest vertices to consider for loss calculation

    Returns:
        torch.Tensor: Contact loss value
    """

    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0])[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0])[:num_vertices]]

    # Calculate pairwise distances between all human and object vertices
    distances = torch.cdist(verts_A.float(), verts_B.float())  # (N, M)

    # Find minimum distance for each vertex in A to any vertex in B
    min_distances_A_to_B = torch.min(distances, dim=1).values  # (N,)

    # Find minimum distance for each vertex in B to any vertex in A
    min_distances_B_to_A = torch.min(distances, dim=0).values  # (M,)

    # Combine all minimum distances
    all_min_distances = torch.cat([min_distances_A_to_B, min_distances_B_to_A], dim=0)

    # Sort and take the smallest distances (closest contacts)
    sorted_min_distances, _ = torch.sort(all_min_distances)

    # Clip negative values (should not happen with squared distances) and take top k
    top_k = min(top_k, len(sorted_min_distances))
    clipped_distances = torch.clamp(sorted_min_distances[:top_k], min=0.0)

    # Apply Huber loss
    contact_loss_value = huber_loss(clipped_distances, delta=delta)

    # Return mean of closest distances as contact loss
    # contact_loss_value = torch.mean(clipped_distances)

    return contact_loss_value


def contact_depth_loss(
    verts_A, verts_B, cameras, num_vertices=2000, top_k=200, delta=0.001, screen_dist_thresh=20.0
):
    """
    Contact loss that only penalizes depth (z) differences.
    First filters object vertices (verts_B) to only those whose 2D screen
    projection is within screen_dist_thresh pixels of the contact body part
    (verts_A), then uses 3D NN pairing on the filtered set and computes loss
    only on view-space z differences.

    Args:
        verts_A (torch.Tensor): Contact body vertices (N, 3) in world space
        verts_B (torch.Tensor): Object vertices (M, 3) in world space
        cameras: PyTorch3D camera object
        num_vertices (int): Max vertices to subsample per point cloud
        top_k (int): Number of closest pairs to use for loss
        delta (float): Huber loss threshold
        screen_dist_thresh (float): Max pixel distance in 2D for an object
            vertex to be considered overlapping with the contact body part
    """
    if verts_A.shape[0] == 0 or verts_B.shape[0] == 0:
        return torch.tensor(0.0, device=verts_A.device, requires_grad=True)

    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0], device=verts_A.device)[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0], device=verts_B.device)[:num_vertices]]

    # Project to screen space for 2D filtering
    screen_A = cameras.transform_points_screen(verts_A.unsqueeze(0)).squeeze(0)[:, :2]  # (N, 2)
    screen_B = cameras.transform_points_screen(verts_B.unsqueeze(0)).squeeze(0)[:, :2]  # (M, 2)

    # For each object vertex, compute min 2D distance to any contact body vertex
    dists_2d = torch.cdist(screen_B.float(), screen_A.float())  # (M, N)
    min_dists_2d = dists_2d.min(dim=1).values  # (M,)

    # Keep only object vertices within the 2D threshold
    mask = min_dists_2d < screen_dist_thresh
    if mask.sum() == 0:
        return torch.tensor(0.0, device=verts_A.device, requires_grad=True)
    verts_B = verts_B[mask]

    # 3D nearest-neighbor pairing on filtered set
    distances_3d = torch.cdist(verts_A.float(), verts_B.float())  # (N, M')
    nn_idx_A = distances_3d.argmin(dim=1)  # (N,) each A -> nearest filtered B
    nn_idx_B = distances_3d.argmin(dim=0)  # (M',) each filtered B -> nearest A

    # Project to view space to get depth (z)
    view_transform = cameras.get_world_to_view_transform()
    z_A = view_transform.transform_points(verts_A.unsqueeze(0)).squeeze(0)[:, 2]  # (N,)
    z_B = view_transform.transform_points(verts_B.unsqueeze(0)).squeeze(0)[:, 2]  # (M',)

    # Depth differences for paired vertices
    depth_diff_A = (z_A - z_B[nn_idx_A]).abs()  # (N,)
    depth_diff_B = (z_B - z_A[nn_idx_B]).abs()  # (M',)

    all_depth_diffs = torch.cat([depth_diff_A, depth_diff_B], dim=0)
    sorted_diffs, _ = all_depth_diffs.sort()

    top_k = min(top_k, len(sorted_diffs))
    return huber_loss(sorted_diffs[:top_k], delta=delta)


def contact_smoothness_loss(verts_A_seq, verts_B_seq, top_k=200):
    """
    Calculate contact smoothness loss by enforcing constant distances for contact point pairs.

    This loss identifies the top k closest vertex pairs at the middle frame and penalizes
    temporal variations in their pairwise distances, encouraging stable contact relationships.

    Args:
        verts_A_seq (list or torch.Tensor): Sequence of vertices for object A, shape (T, N_A, 3)
        verts_B_seq (list or torch.Tensor): Sequence of vertices for object B, shape (T, N_B, 3)
        top_k (int): Number of closest point pairs to track

    Returns:
        torch.Tensor: Mean absolute deviation from constant distance (contact variance loss)
    """
    window_size = len(verts_A_seq)

    if window_size < 2:
        device = verts_A_seq[0].device if window_size > 0 else torch.device("cpu")
        return torch.tensor(0.0, device=device, requires_grad=True)

    verts_A_middle = verts_A_seq[window_size // 2]
    verts_B_middle = verts_B_seq[window_size // 2]

    # Find the top k closest points between verts_A_middle and verts_B_middle
    # Calculate pairwise distances at the middle frame
    distances_middle = torch.cdist(verts_A_middle.float(), verts_B_middle.float())  # (N_A, N_B)

    # Flatten distances and find top k smallest
    distances_flat = distances_middle.flatten()
    top_k = min(top_k, len(distances_flat))
    top_k_values, top_k_indices = torch.topk(distances_flat, top_k, largest=False)

    # Convert flat indices back to (i, j) pairs
    num_B = verts_B_middle.shape[0]
    idx_A = top_k_indices // num_B  # Indices in verts_A
    idx_B = top_k_indices % num_B  # Indices in verts_B

    # Track distances for these point pairs across the entire sequence
    distance_sequences = []
    for t in range(window_size):
        verts_A_t = verts_A_seq[t]
        verts_B_t = verts_B_seq[t]

        # Get the selected vertices
        selected_A = verts_A_t[idx_A]  # (top_k, 3)
        selected_B = verts_B_t[idx_B]  # (top_k, 3)

        # Calculate distances for these pairs
        distances_t = torch.norm(selected_A - selected_B, dim=-1)  # (top_k,)
        distance_sequences.append(distances_t)

    # Stack into tensor: (window_size, top_k) where each column is a point pair's distance trajectory
    distance_sequences = torch.stack(distance_sequences, dim=0)

    # Compute temporal mean distance for each point pair
    avg_distance_per_pair = torch.mean(distance_sequences, dim=0)  # (top_k,)

    # Penalize deviation from constant distance (temporal variance)
    diff_from_mean = torch.abs(
        distance_sequences - avg_distance_per_pair.unsqueeze(0)
    )  # (window_size, top_k)

    # Return mean absolute deviation across all timesteps and point pairs
    return torch.mean(diff_from_mean)


def contact_distribution_smoothness_loss(
    human_contact_verts_seq, obj_verts_seq, temperature=100.0, num_obj_verts=2000
):
    """
    Penalize rapid changes in the contact region on the object surface.

    Computes a soft contact distribution over object vertex *indices* at each
    frame (which vertices are closest to the human contact part).  Because
    vertex indices map to fixed surface locations on a rigid object, the
    distribution is implicitly in object-local space — object translation and
    rotation do not cause spurious penalties.

    Gradual contact migration (re-gripping, sliding) is allowed; only sudden
    frame-to-frame jumps in the contact region are penalised.

    Args:
        human_contact_verts_seq: (T, N, 3) human contact vertices per frame.
        obj_verts_seq: (T, M, 3) object vertices per frame.
        temperature: Softmax sharpness — higher = more focused on closest verts.
        num_obj_verts: Subsample object verts to this count for efficiency.
            A fixed, evenly-spaced subsample is used so the same surface
            points are compared across frames.
    """
    T = len(human_contact_verts_seq)
    if T < 2:
        dev = human_contact_verts_seq[0].device if T > 0 else torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=True)

    obj_verts = obj_verts_seq  # (T, M, 3)
    if obj_verts.shape[1] > num_obj_verts:
        subsample_idx = torch.linspace(0, obj_verts.shape[1] - 1, num_obj_verts).long()
        obj_verts = obj_verts[:, subsample_idx]  # (T, M', 3)

    dists = torch.cdist(obj_verts.float(), human_contact_verts_seq.float())  # (T, M', N)
    min_dists = torch.min(dists, dim=2).values  # (T, M')
    distributions = torch.softmax(-min_dists * temperature, dim=1)  # (T, M')
    diff = distributions[1:] - distributions[:-1]  # (T-1, M')
    return diff.pow(2).sum(dim=-1).mean()


def contact_center_loss(verts_A, verts_B):
    """
    Calculate contact center loss based on the center of the two point clouds.
    """
    center_A = torch.mean(verts_A, dim=0)  # (3,)
    center_B = torch.mean(verts_B, dim=0)  # (3,)
    return torch.nn.functional.l1_loss(center_A, center_B)


def ground_loss(verts_seq, gravity_axis="z", height=0.14):
    # verts_seq
    if gravity_axis == "y":
        min_verts_seq = torch.min(verts_seq[:, :, 1], dim=1).values
    elif gravity_axis == "z":
        min_verts_seq = torch.min(verts_seq[:, :, 2], dim=1).values
    else:
        raise ValueError(f"Invalid gravity axis: {gravity_axis}")

    return torch.mean(torch.abs(min_verts_seq - height))


def penetration_loss(verts_A, sdf_B, threshold=0.0):
    """
    Compute penetration loss to prevent interpenetration between meshes.

    Penalizes vertices of A that penetrate into B (SDF < threshold).

    Args:
        verts_A: Vertices of object A (N, 3) - e.g., human mesh vertices
        sdf_B: Pre-computed SDF of object B (MeshSDF object)
        threshold: Distance threshold in meters (default 0.0 = at surface)
                   - threshold=0.0: Penalize penetration (SDF < 0)
                   - threshold>0.0: Penalize being too close (SDF < threshold)
                   - threshold<0.0: Allow some penetration

    Returns:
        Penetration loss value (scalar)
    """
    # Query SDF at vertices of A
    sdf_values = sdf_B.query(verts_A, method="grid")

    # Penetration occurs when SDF < threshold
    # For threshold=0: negative values = inside mesh (penetrating)
    penetration = threshold - sdf_values
    penetration = torch.clamp(penetration, min=0.0)  # Only penalize violations

    # Return mean penetration
    return penetration.sum()


def keypoint_loss(pred_keypoints, gt_keypoints, gt_conf=None, conf_thres=0.6):
    if gt_conf is not None:
        gt_keypoints = gt_keypoints[gt_conf > conf_thres]
        pred_keypoints = pred_keypoints[gt_conf > conf_thres]

    if pred_keypoints.shape[0] == 0:
        return pred_keypoints.sum() * 0.0

    return torch.nn.functional.l1_loss(pred_keypoints, gt_keypoints)


def smoothness_loss(seq, beta=1.0):
    first_order_loss = torch.nn.functional.l1_loss(seq[1:] - seq[:-1], torch.zeros_like(seq[1:]))
    second_order_loss = torch.nn.functional.l1_loss(
        seq[2:] - 2 * seq[1:-1] + seq[:-2],
        torch.zeros_like(seq[2:]),
    )
    return first_order_loss + beta * second_order_loss


def pose_smoothness_loss(pose_seq, beta=1.0, joint_weights=None):
    """
    Pose smoothness loss using geodesic distance on SO(3) manifold

    Args:
        pose_seq: (frame_num, N, 3) - axis-angle pose sequence
        beta: Overall loss weight scaling factor
        joint_weights: (N,) - Per-joint importance weights
    """
    from pytorch3d.transforms import axis_angle_to_matrix, so3_log_map

    if pose_seq.dim() != 3:
        raise ValueError(
            f"Expected pose_seq to have 3 dimensions (frame_num, N, 3), got {pose_seq.shape}"
        )

    frame_num, num_joints, _ = pose_seq.shape
    device = pose_seq.device

    if frame_num < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Convert axis-angle to rotation matrices using PyTorch3D
    pose_flat = pose_seq.reshape(-1, 3)  # (frame_num * N, 3)
    rot_mats = axis_angle_to_matrix(pose_flat)  # (frame_num * N, 3, 3)
    rot_mats = rot_mats.reshape(frame_num, num_joints, 3, 3)  # (frame_num, N, 3, 3)

    # Set default joint weights
    if joint_weights is None:
        joint_weights = torch.ones(num_joints, device=device)
    else:
        joint_weights = joint_weights.to(device)

    # ===== Vectorized Velocity Computation =====
    # Compute all relative rotations at once
    R_t = rot_mats[:-1]  # (frame_num-1, N, 3, 3)
    R_t_plus_1 = rot_mats[1:]  # (frame_num-1, N, 3, 3)

    # Compute relative rotations: R_t^{-1} R_{t+1}
    R_t_inv = R_t.transpose(-2, -1)  # (frame_num-1, N, 3, 3)
    R_rel = torch.matmul(R_t_inv, R_t_plus_1)  # (frame_num-1, N, 3, 3)

    # Flatten for batch processing with so3_log_map
    R_rel_flat = R_rel.reshape(-1, 3, 3)  # ((frame_num-1)*N, 3, 3)
    log_R_rel_flat = so3_log_map(R_rel_flat)  # ((frame_num-1)*N, 3)

    # Reshape back to get velocity vectors
    log_velocities = log_R_rel_flat.reshape(frame_num - 1, num_joints, 3)  # (frame_num-1, N, 3)

    # ===== Velocity Smoothness Loss =====
    # L_vel = Σ_{t,j} w_j ||log(R_{t,j}^{-1} R_{t+1,j})||^2
    vel_norms = torch.norm(log_velocities, dim=-1)  # (frame_num-1, N)
    weighted_vel_norms = vel_norms**2 * joint_weights.unsqueeze(0)  # (frame_num-1, N)
    vel_loss = weighted_vel_norms.sum() / (frame_num - 1)  # Average over time

    # ===== Acceleration Smoothness Loss =====
    acc_loss = torch.tensor(0.0, device=device)

    if frame_num >= 3:  # Need at least 3 frames for acceleration
        # L_acc = Σ_{t,j} w_j ||v_{t+1,j} - v_{t,j}||^2
        # Compute acceleration vectors: v_{t+1} - v_t
        accelerations = log_velocities[1:] - log_velocities[:-1]  # (frame_num-2, N, 3)

        # Compute acceleration magnitudes
        acc_norms = torch.norm(accelerations, dim=-1)  # (frame_num-2, N)
        weighted_acc_norms = acc_norms**2 * joint_weights.unsqueeze(0)  # (frame_num-2, N)
        acc_loss = weighted_acc_norms.sum() / (frame_num - 2)  # Average over time

    # Combine velocity and acceleration losses
    return vel_loss + beta * acc_loss


def reg_loss(new_seq, old_seq, use_l2=False):
    if new_seq.shape != old_seq.shape:
        raise ValueError(
            f"Shape mismatch: new_seq.shape={new_seq.shape}, old_seq.shape={old_seq.shape}"
        )
    if use_l2:
        return torch.nn.functional.mse_loss(new_seq, old_seq)
    else:
        return torch.nn.functional.l1_loss(new_seq, old_seq)
