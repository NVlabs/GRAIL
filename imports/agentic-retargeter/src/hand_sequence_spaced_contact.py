"""Select distinct, spatially separated pad samples for a contact objective.

This is an opt-in tensor helper, not a material-anchor extractor or a motion
acceptance check. Distances and point identities come from the caller's current
pose. Selection is discrete and detached; the gathered original distances keep
their gradients. No mesh queries or geometry changes occur here.
"""
import math


def select_spaced_contact(distances, hand_points, *, count=4,
                          min_spacing_m=.003, object_points=None,
                          eligible_mask=None, contact_mask=None):
    """Greedily select the smallest distance, then exclude nearby points.

    Args:
        distances: Nonnegative floating tensor [B,N], normally absolute surface
            distance. Nonfinite/negative entries are ineligible, not repaired.
        hand_points: Floating [B,N,3] points in one common metric frame per row.
        count: Desired number of unique point identities per active row.
        min_spacing_m: Minimum Euclidean spacing, inclusive at the boundary.
        object_points: Optional true closest-object points [B,N,3], paired with
            the same input identities. If absent, only robot spacing is checked.
        eligible_mask: Optional bool [B,N]; no distance cutoff is applied here.
        contact_mask: Optional bool [B]. Inactive rows select nothing and have
            zero missing_count; coverage_complete is false for inactive rows.

    Returns a dictionary of tensors on the original device. ``indices`` has
    -1 in missing slots; ``valid`` distinguishes these from actual samples.
    Missing distances are zero placeholders and MUST be masked for objectives.
    ``valid_distance_mean`` divides by selected_count, never the requested count.
    ``coverage_complete`` means only that all requested spaced points exist;
    distances may still be large. It does not certify contact, collision or
    accepted motion. With no valid points the differentiable sum/mean are zero
    and missing_count explicitly records the deficit. Callers must retain it.

    Ties select the lowest original index. This greedy set is deterministic but
    need not be a maximal-cardinality set. Index selection is not differentiable.
    """
    import torch
    if not isinstance(distances, torch.Tensor) or distances.ndim != 2:
        raise ValueError('distances must be a tensor with shape [B,N]')
    if not distances.is_floating_point():
        raise ValueError('distances must have a floating dtype')
    if type(count) is not int or count <= 0:
        raise ValueError('count must be a positive integer')
    if type(min_spacing_m) not in (int, float) or not math.isfinite(min_spacing_m) or min_spacing_m < 0:
        raise ValueError('min_spacing_m must be finite and nonnegative')
    batch, samples = distances.shape

    def points(value, name):
        if (not isinstance(value, torch.Tensor) or value.shape != (batch, samples, 3)
                or not value.is_floating_point() or value.device != distances.device):
            raise ValueError(name+' must be floating [B,N,3] on the distances device')

    def mask(value, shape, name):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.bool or value.device != distances.device):
            raise ValueError(name+' must be a bool tensor with the declared shape/device')

    points(hand_points, 'hand_points')
    if object_points is not None:
        points(object_points, 'object_points')
    if eligible_mask is not None:
        mask(eligible_mask, (batch, samples), 'eligible_mask')
    if contact_mask is not None:
        mask(contact_mask, (batch,), 'contact_mask')
    device = distances.device
    with torch.no_grad():
        active = (torch.ones(batch, dtype=torch.bool, device=device)
                  if contact_mask is None else contact_mask.detach().clone())
        finite_distance = torch.isfinite(distances) & (distances >= 0)
        finite_geometry = torch.isfinite(hand_points).all(-1)
        if object_points is not None:
            finite_geometry &= torch.isfinite(object_points).all(-1)
        permitted = (torch.ones_like(finite_distance) if eligible_mask is None
                     else eligible_mask.detach())
        available = finite_distance & finite_geometry & permitted & active[:, None]
        initial_eligible_count = available.sum(-1)
        invalid_input_count = ((~finite_distance | ~finite_geometry)
                               & permitted & active[:, None]).sum(-1)
        indices = torch.full((batch, count), -1, dtype=torch.long, device=device)
        valid = torch.zeros((batch, count), dtype=torch.bool, device=device)
        # Keep half-precision geometry from underflowing at millimetre scale.
        hp = hand_points.detach()
        hp = hp.float() if hp.dtype in (torch.float16, torch.bfloat16) else hp
        op = None if object_points is None else object_points.detach()
        if op is not None and op.dtype in (torch.float16, torch.bfloat16):
            op = op.float()
        rows = torch.arange(batch, device=device)
        sample_ids = torch.arange(samples, device=device)
        spacing_squared = float(min_spacing_m)**2
        if samples:
            for slot in range(count):
                ranked = distances.detach().masked_fill(~available, float('inf'))
                selected = ranked.argmin(-1)  # first index wins exact ties
                exists = available[rows, selected]
                indices[:, slot] = torch.where(exists, selected, -1)
                valid[:, slot] = exists
                delta = hp - hp[rows, selected][:, None]
                separated = delta.square().sum(-1) >= spacing_squared
                if op is not None:
                    delta_object = op - op[rows, selected][:, None]
                    separated &= delta_object.square().sum(-1) >= spacing_squared
                available &= (~exists[:, None] | separated)
                # Remove the identity even at zero spacing. Keep the update
                # entirely on device: scalar advanced-index assignment can
                # copy its Python scalar through the host during CUDA capture.
                available &= sample_ids[None, :] != selected[:, None]
        selected_count = valid.sum(-1)
        missing_count = torch.where(active, count-selected_count, 0)
        coverage_complete = active & (selected_count == count)

    # Only selection is detached. These are the caller's actual distances.
    if samples:
        gathered = distances.gather(1, indices.clamp_min(0))
        selected_distances = torch.where(valid, gathered, torch.zeros_like(gathered))
    else:
        # Preserve a valid autograd path even for empty [B,0] source tensors.
        selected_distances = distances.sum(-1, keepdim=True).expand(batch, count)
    distance_sum = selected_distances.sum(-1)
    distance_mean = distance_sum / selected_count.clamp_min(1).to(distances.dtype)
    return dict(indices=indices, valid=valid, distances=selected_distances,
                selected_count=selected_count, missing_count=missing_count,
                coverage_complete=coverage_complete, active=active,
                initial_eligible_count=initial_eligible_count,
                invalid_input_count=invalid_input_count,
                valid_distance_sum=distance_sum, valid_distance_mean=distance_mean,
                spacing_scope='robot_and_true_object_points' if object_points is not None else 'robot_points_only',
                min_spacing_m=float(min_spacing_m), requested_count=count,
                distance_threshold_m=None, selection_differentiable=False,
                greedy_selection_is_maximum_cardinality=False,
                creates_fixed_material_anchors=False, motion_accepted=False)
