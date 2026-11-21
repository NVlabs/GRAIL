"""Native foot surfaces against an explicitly supplied world-space floor plane."""

import json
import math
from collections.abc import Mapping

import torch

FOOT_PARTS = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")


def foot_region_indices(human_model):
    """Resolve native heel/foot and toe surfaces; never substitute joint centers.

    SOMA's ankle segment includes descendants, including the toe. Remove those
    toe IDs from the ankle region so toe support does not also pin the heel.
    """
    regions = []
    for label in FOOT_PARTS:
        raw = torch.as_tensor(human_model.get_segment_indices([label])).reshape(-1)
        if raw.numel() == 0 or raw.is_floating_point() or raw.dtype == torch.bool:
            raise ValueError(f"Foot region {label} requires nonempty integer native vertex IDs")
        regions.append(torch.unique(raw.detach().to(dtype=torch.long, device="cpu")))
    for heel, toe in ((0, 1), (2, 3)):
        regions[heel] = regions[heel][~torch.isin(regions[heel], regions[toe])]
        if regions[heel].numel() == 0:
            raise ValueError(f"Foot region {FOOT_PARTS[heel]} has no vertices outside its toe")
    return tuple(regions)


def normalized_ground_plane(plane, reference):
    """Return detached unit normal and offset for ``normal @ x + offset_m = 0``.

    The normal must point into free space (up). Normal and offset are normalized
    together. No floor is inferred from the candidate human or a default height.
    """
    if not isinstance(plane, Mapping) or "normal" not in plane or "offset_m" not in plane:
        raise ValueError("human_foot_ground requires an explicit plane with normal and offset_m")
    source = plane.get("source")
    if not (
        (isinstance(source, str) and source.strip()) or (isinstance(source, Mapping) and source)
    ):
        raise ValueError("Ground plane requires a nonempty source provenance")
    try:
        json.dumps(source, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ground plane source provenance must be JSON serializable") from exc
    normal = torch.as_tensor(plane["normal"], dtype=reference.dtype, device=reference.device)
    offset = torch.as_tensor(plane["offset_m"], dtype=reference.dtype, device=reference.device)
    if normal.shape != (3,) or offset.ndim != 0:
        raise ValueError("Ground plane normal must have shape [3] and offset_m must be scalar")
    if not bool(torch.isfinite(normal).all()) or not bool(torch.isfinite(offset)):
        raise ValueError("Ground plane coefficients must be finite")
    norm = normal.norm()
    if norm.item() <= 1e-8:
        raise ValueError("Ground plane normal must be nonzero")
    return (normal / norm).detach(), (offset / norm).detach()


def foot_ground_terms(
    vertices,
    regions,
    probabilities,
    *,
    plane,
    support_threshold=0.5,
    robust_delta_m=0.005,
    support_height_m=0.0,
    penetration_margin_m=0.0,
):
    """Return nonpenetration and contact-gated sole-height penalties in metres.

    Nonpenetration evaluates all native foot vertices in every frame and
    penalizes the deepest one, preventing sparse penetration from being diluted
    by the many vertices above the floor. Support uses each heel/toe's lower
    surface envelope; it does not flatten the foot volume onto the floor.
    ``amin`` distributes gradients across tied minima and is piecewise smooth;
    a Huber penalty limits sensitivity away from contact. These are soft losses,
    not a hard guarantee that a finite-weight fitted mesh cannot penetrate.
    An explicit nonnegative ``penetration_margin_m`` moves the soft barrier
    above the measured floor without changing that plane or its provenance.
    It cannot exceed the supported sole target, to avoid conflicting targets.

    Fixed GEM labels are [L ankle, L toe, R ankle, R toe], independently gated.
    Unsupported parts can lift freely; they still cannot pass through the floor
    without a nonpenetration penalty. Existing horizontal stance losses remain
    separate. Missing labels disable height attraction only, never penetration.
    """
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not vertices.shape[0]:
        raise ValueError("Foot ground requires native vertices [T,V,3] with T > 0")
    if len(regions) != 4:
        raise ValueError("Foot ground requires four native heel/toe regions")
    if not math.isfinite(support_threshold) or not 0 <= support_threshold <= 1:
        raise ValueError("Foot ground support_threshold must be in [0,1]")
    if not math.isfinite(robust_delta_m) or robust_delta_m <= 0:
        raise ValueError("Foot ground robust_delta_m must be finite and positive")
    if not math.isfinite(support_height_m) or support_height_m < 0:
        raise ValueError("Foot ground support_height_m must be finite and nonnegative")
    if (
        not math.isfinite(penetration_margin_m)
        or penetration_margin_m < 0
        or penetration_margin_m > support_height_m
    ):
        raise ValueError(
            "Foot ground penetration_margin_m must be finite and in [0, support_height_m]"
        )
    normal, offset = normalized_ground_plane(plane, vertices)
    low_surfaces = []
    for label, ids in zip(FOOT_PARTS, regions):
        ids = torch.as_tensor(ids, device=vertices.device)
        if ids.ndim != 1 or not ids.numel() or ids.is_floating_point() or ids.dtype == torch.bool:
            raise ValueError(f"Foot region {label} requires nonempty integer vertex IDs")
        if bool((ids < 0).any()) or bool((ids >= vertices.shape[1]).any()):
            raise ValueError(f"Foot region {label} contains out-of-range native vertex IDs")
        heights = vertices[:, ids.long()] @ normal + offset
        low_surfaces.append(heights.amin(dim=1))
    low_surfaces = torch.stack(low_surfaces, dim=1)

    active = None
    if probabilities is not None:
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != vertices.shape[0]
            or probabilities.shape[1] < 4
        ):
            raise ValueError("Foot ground support probabilities require shape [T,>=4]")
        if not bool(torch.isfinite(probabilities[:, :4]).all()):
            raise ValueError("Foot ground support probabilities must be finite")
        active = probabilities[:, :4].detach().to(vertices.device) > support_threshold
    return _surface_terms(
        low_surfaces, active, robust_delta_m, support_height_m, penetration_margin_m
    )


def _surface_terms(low_surfaces, active, delta, support_height, penetration_margin):
    def robust(value):
        magnitude = value.abs()
        return torch.where(magnitude < delta, 0.5 * value.square() / delta, magnitude - 0.5 * delta)

    penetration = robust((penetration_margin - low_surfaces.amin(dim=1)).clamp_min(0.0)).mean()
    support = low_surfaces.sum() * 0.0
    if active is not None:
        support = (robust(low_surfaces - support_height) * active).sum() / active.sum().clamp_min(1)
    return {"penetration": penetration, "support_height": support}


class FootGroundConstraint:
    """Validate fixed plane, native indices and support labels once per fit.

    Calls evaluate only tensor expressions without per-iteration host transfers
    or candidate-dependent support classification.
    """

    def __init__(
        self,
        vertices,
        regions,
        probabilities,
        *,
        plane,
        support_threshold=0.5,
        robust_delta_m=0.005,
        support_height_m=0.0,
        penetration_margin_m=0.0,
    ):
        # The one-time functional evaluation provides the same validation as the
        # stateless API; no retained graph from the input state is required.
        with torch.no_grad():
            foot_ground_terms(
                vertices,
                regions,
                probabilities,
                plane=plane,
                support_threshold=support_threshold,
                robust_delta_m=robust_delta_m,
                support_height_m=support_height_m,
                penetration_margin_m=penetration_margin_m,
            )
        self.shape = vertices.shape
        self.normal, self.offset = normalized_ground_plane(plane, vertices)
        self.regions = tuple(
            torch.as_tensor(ids, device=vertices.device, dtype=torch.long) for ids in regions
        )
        self.active = (
            None
            if probabilities is None
            else probabilities[:, :4].detach().to(vertices.device) > support_threshold
        )
        self.delta = robust_delta_m
        self.support_height = support_height_m
        self.penetration_margin = penetration_margin_m

    def terms(self, vertices):
        if (
            vertices.shape != self.shape
            or vertices.device != self.normal.device
            or vertices.dtype != self.normal.dtype
        ):
            raise ValueError("Foot ground geometry layout/device/dtype changed within a fit")
        low_surfaces = torch.stack(
            [(vertices[:, ids] @ self.normal + self.offset).amin(dim=1) for ids in self.regions],
            dim=1,
        )

        return _surface_terms(
            low_surfaces, self.active, self.delta, self.support_height, self.penetration_margin
        )
