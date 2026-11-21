"""Small geometric losses inspired by ObjRetarget equations 10--13.

These are optional proposal priors, not contact or collision acceptance metrics.
The paper's named edge-length term is actually an edge-vector difference; this
implementation preserves that distinction and never rescales robot or object.
"""


def tetrahedral_vector_loss(vertices, reference, weights=None):
    """Return [B] edge-vector loss for [B,F,4,3] tetrahedral vertices.

    Suggested declared vertex order is palm, fingertip, adjacent fingertip,
    object-region contact representative. Reference is the same shape or
    [F,4,3]; weights, when supplied, are nonnegative [F] or [B,F]. All six
    tetrahedron edges are included. Vertices must share one coordinate frame.
    """
    import torch

    if (not isinstance(vertices, torch.Tensor) or vertices.ndim != 4
            or vertices.shape[-2:] != (4, 3) or min(vertices.shape[:2]) < 1
            or not vertices.is_floating_point() or not torch.isfinite(vertices).all()):
        raise ValueError("vertices must be finite floating [B,F,4,3]")
    if (not isinstance(reference, torch.Tensor)
            or reference.shape not in (vertices.shape, vertices.shape[1:])
            or reference.device != vertices.device or reference.dtype != vertices.dtype
            or not torch.isfinite(reference).all()):
        raise ValueError("reference must match vertices or [F,4,3] with same device/dtype")
    if weights is not None:
        if (not isinstance(weights, torch.Tensor)
                or weights.shape not in (vertices.shape[:2], vertices.shape[1:2])
                or weights.device != vertices.device or weights.dtype != vertices.dtype
                or not torch.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError("weights must be nonnegative finite [F] or [B,F]")
    delta = vertices - reference
    edges = torch.stack([delta[..., i, :] - delta[..., j, :]
                         for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]], -2)
    per_finger = edges.square().sum((-1, -2))
    return (per_finger if weights is None else per_finger * weights).sum(-1)


def palm_relative_contact_loss(palm, rotation, contact, reference_relative, weights=None):
    """Return [B] error of [B,F,3] contacts in a [B,3,3] palm frame.

    ``rotation`` maps palm-local vectors into the common coordinate frame.
    ``reference_relative`` is [B,F,3] or [F,3] and has already been expressed in
    its reference palm frame. Rotation construction/validity is caller-owned.
    No normals or geometry scaling are introduced.
    """
    import torch

    if (not isinstance(contact, torch.Tensor) or contact.ndim != 3
            or contact.shape[-1] != 3 or min(contact.shape[:2]) < 1
            or not contact.is_floating_point() or not torch.isfinite(contact).all()):
        raise ValueError("contact must be finite floating [B,F,3]")
    shapes = [("palm", palm, (contact.shape[0], 3)),
              ("rotation", rotation, (contact.shape[0], 3, 3))]
    for name, value, shape in shapes:
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != contact.device or value.dtype != contact.dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} has invalid shape, values, device, or dtype")
    if (not isinstance(reference_relative, torch.Tensor)
            or reference_relative.shape not in (contact.shape, contact.shape[1:])
            or reference_relative.device != contact.device
            or reference_relative.dtype != contact.dtype
            or not torch.isfinite(reference_relative).all()):
        raise ValueError("reference_relative must be finite [B,F,3] or [F,3]")
    local = torch.einsum("bji,bfj->bfi", rotation, contact-palm[:, None])
    per_contact = (local-reference_relative).square().sum(-1)
    if weights is not None:
        if (not isinstance(weights, torch.Tensor)
                or weights.shape not in (contact.shape[:2], contact.shape[1:2])
                or weights.device != contact.device or weights.dtype != contact.dtype
                or not torch.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError("weights must be nonnegative finite [F] or [B,F]")
        per_contact = per_contact * weights
    return per_contact.sum(-1)
