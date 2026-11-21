"""Optional authored-contact surface-normal preference, using original topology.

Outward normal directions follow mesh winding; this module never repairs winding,
selects a different material face, or searches for nearby contact points. Opposing
normals do not establish collision clearance, grasp stability, or force closure.
Used only when the reviewed acquisition normal preference is explicitly enabled.
"""

import math
import numbers
from collections.abc import Mapping

import torch


def _geometry(vertices, name):
    if not torch.is_tensor(vertices) or vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"{name} must be a [T,V,3] tensor")
    if vertices.shape[0] < 1 or vertices.shape[1] < 1:
        raise ValueError(f"{name} needs nonempty time and vertex dimensions")
    # Float32/64 avoid half-precision triangle-area underflow on small meshes.
    if vertices.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if not torch.isfinite(vertices).all():
        raise ValueError(f"{name} must be finite")


def _indices(value, name, device, upper, dimensions):
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be integer indices") from exc
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    # Empty Python lists default to float32; they contain no invalid indices.
    if tensor.numel() and tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"{name} must be integer indices")
    tensor = tensor.to(dtype=torch.long)
    if tensor.numel() and ((tensor < 0).any() or (tensor >= upper).any()):
        raise ValueError(f"{name} contains an out-of-range index")
    return tensor


def _faces(faces, vertices, name):
    result = _indices(faces, name, vertices.device, vertices.shape[1], 2)
    if result.shape[1] != 3:
        raise ValueError(f"{name} must contain triangles")
    return result


def _unit(vectors, name):
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if not torch.isfinite(lengths).all() or (lengths <= torch.finfo(vectors.dtype).tiny).any():
        raise ValueError(f"{name} has a degenerate or nonfinite normal")
    return vectors / lengths


def local_vertex_normals(human, human_faces, vertex_ids):
    """Return [T,P,3] area-weighted normals at the authored human vertices.

    Adjacent zero-area triangles contribute zero. A vertex with no incident
    triangles, no remaining area, or cancelling incident normals is rejected.
    Faces and ids are discrete; gradients flow through the vertex coordinates.
    """
    _geometry(human, "human")
    faces = _faces(human_faces, human, "human_faces")
    ids = _indices(vertex_ids, "vertex_ids", human.device, human.shape[1], 1)
    if ids.numel() == 0:
        return human[:, :0]
    normals = []
    for vertex_id in ids:
        adjacent = faces[(faces == vertex_id).any(dim=-1)]
        if len(adjacent) == 0:
            raise ValueError("An authored human vertex has no incident triangles")
        tri = human[:, adjacent]
        area_vectors = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
        vector = area_vectors.sum(dim=1)
        length = torch.linalg.vector_norm(vector, dim=-1)
        area_sum = torch.linalg.vector_norm(area_vectors, dim=-1).sum(dim=-1)
        # Opposite coincident faces or invalid winding may cancel the normal.
        # Do not quietly replace it with an arbitrary axis or epsilon-normal.
        if (length <= area_sum * (8 * torch.finfo(human.dtype).eps)).any():
            raise ValueError("Incident human triangle normals cancel or have zero area")
        normals.append(_unit(vector, "Human vertex"))
    return torch.stack(normals, dim=1)


def _field(pair, name):
    try:
        return pair[name] if isinstance(pair, Mapping) else getattr(pair, name)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Contact pair is missing {name}") from exc


def _pairs(pairs):
    if not isinstance(pairs, (tuple, list)):
        raise ValueError("pairs must be an explicit list or tuple of authored correspondences")
    human_ids, face_ids, weights = [], [], []
    for pair in pairs:
        human_id, face_id = _field(pair, "human_vertex_id"), _field(pair, "object_face_id")
        for index in (human_id, face_id):
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                raise ValueError("Authored human vertex and material face ids must be integers")
        weight = _field(pair, "weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, numbers.Real)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Pair weights must be positive and finite")
        human_ids.append(int(human_id))
        face_ids.append(int(face_id))
        weights.append(float(weight))
    return human_ids, face_ids, weights


def material_normal_dots(human, objects, human_faces, object_faces, pairs):
    """Return [T,P] human/material outward-normal dots for fixed authored pairs.

    A triangle has a constant face normal, so its contact barycentric coordinate
    does not affect the material normal. Vertex/face ids are never changed.
    """
    _geometry(human, "human")
    _geometry(objects, "objects")
    if (
        human.shape[0] != objects.shape[0]
        or human.device != objects.device
        or human.dtype != objects.dtype
    ):
        raise ValueError("Human/object geometry must share timeline, floating dtype and device")
    human_ids, face_ids, _ = _pairs(pairs)
    faces = _faces(object_faces, objects, "object_faces")
    chosen = _indices(face_ids, "object_face_ids", human.device, len(faces), 1)
    normal_human = local_vertex_normals(human, human_faces, human_ids)
    if not pairs:
        return (human[:, :0] + objects[:, :0]).sum(dim=-1)
    tri = objects[:, faces[chosen]]
    area_vectors = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
    normal_object = _unit(area_vectors, "Authored object triangle")
    return (normal_human * normal_object).sum(dim=-1).clamp(-1.0, 1.0)


def normal_opposition_penalty(dots, weights=None, opposition_cos=0.2):
    """Half weighted mean plus half worst ReLU(dot + opposition_cos)^2.

    The time axis includes exactly the caller's selected contact interval. This
    helper does not decide contact timing. Equal positive weights are the default.
    An exactly parallel configuration can have zero angular gradient; this soft
    preference is not a replacement for pose initialization or collision checks.
    """
    if not torch.is_tensor(dots) or dots.ndim != 2 or dots.shape[0] < 1:
        raise ValueError("dots must be a [T,P] tensor with nonempty timeline")
    if dots.dtype not in (torch.float32, torch.float64) or not torch.isfinite(dots).all():
        raise ValueError("dots must be finite float32 or float64")
    if (dots.abs() > 1 + 8 * torch.finfo(dots.dtype).eps).any():
        raise ValueError("Normal dots must lie within [-1,1]")
    if (
        isinstance(opposition_cos, bool)
        or not isinstance(opposition_cos, numbers.Real)
        or not math.isfinite(opposition_cos)
        or not 0 <= opposition_cos <= 1
    ):
        raise ValueError("opposition_cos must be finite in [0,1]")
    if weights is None:
        weights = [1.0] * dots.shape[1]
    if not isinstance(weights, (tuple, list)) or len(weights) != dots.shape[1]:
        raise ValueError("weights must contain one explicit scalar per pair")
    if any(
        isinstance(w, bool) or not isinstance(w, numbers.Real) or not math.isfinite(w) or w <= 0
        for w in weights
    ):
        raise ValueError("Normal weights must be positive and finite")
    if not weights:
        return dots.sum() * 0
    largest = max(weights)
    normalized = dots.new_tensor([w / largest for w in weights])
    violation = torch.relu(dots + opposition_cos).square()
    mean = (violation * normalized).sum() / (dots.shape[0] * normalized.sum())
    return 0.5 * (mean + violation.max())


def material_normal_loss(human, objects, human_faces, object_faces, pairs, opposition_cos=0.2):
    """Convenience wrapper for one already-selected contact interval."""
    _, _, weights = _pairs(pairs)
    dots = material_normal_dots(human, objects, human_faces, object_faces, pairs)
    return normal_opposition_penalty(dots, weights, opposition_cos)
