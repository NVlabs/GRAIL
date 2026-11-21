"""Fixed, phase-aware native foot support on finite static scene patches.

This is a surface-column constraint for floors, stair treads and slopes. It is
not a signed mesh distance, a full riser/edge collision check, a swept collision
check or a physical support guarantee. In particular, a vertex outside a
patch's projected footprint receives no penetration penalty from that patch.
The optional column_exit mode uses shortest exit from each convex downward
column, then their maximum: a continuous lower bound on exit distance from
their union, not the union's exact signed distance. Collision columns must
retain overlapping original solid footprints, not merely exposed tread strips.
Support assignments and material IDs are fixed before optimization; they are
never selected from the candidate's nearest surface or its current height.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from grail.optimization.foot_ground import FOOT_PARTS


def _provenance(value, label):
    if not ((isinstance(value, str) and value.strip()) or (isinstance(value, Mapping) and value)):
        raise ValueError(f"{label} requires nonempty source provenance")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} source must be finite JSON data") from exc


def _tensor(value, reference, label, shape=None):
    try:
        result = (
            torch.as_tensor(value, dtype=reference.dtype, device=reference.device).detach().clone()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} requires explicit numeric values") from exc
    if (shape is not None and tuple(result.shape) != shape) or not bool(
        torch.isfinite(result).all()
    ):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _ids(value, reference, label):
    result = torch.as_tensor(value, device=reference.device)
    if (
        result.ndim != 1
        or not result.numel()
        or result.is_floating_point()
        or result.dtype == torch.bool
    ):
        raise ValueError(f"{label} requires nonempty integer native vertex IDs")
    result = torch.unique(result.detach().clone().long())
    if bool((result < 0).any()) or bool((result >= reference.shape[1]).any()):
        raise ValueError(f"{label} contains out-of-range native vertex IDs")
    return result


def _nonnegative(value, label):
    if (
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _phase_weights(reference, count, weight, blend_frames):
    values = reference.new_full((count,), weight)
    if blend_frames:
        phase = torch.arange(1, count + 1, device=reference.device, dtype=reference.dtype)
        ramp = torch.minimum(phase, phase.flip(0)).div(blend_frames + 1).clamp_max(1)
        values = values * ramp.square() * (3 - 2 * ramp)
    return values


@dataclass(frozen=True)
class _Patch:
    origin: torch.Tensor
    normal: torch.Tensor
    polygon: torch.Tensor
    inward: torch.Tensor

    def heights(self, points):
        return (points - self.origin) @ self.normal

    def edge_distances(self, points):
        # Inward edge vectors lie in the patch plane: no explicit projection
        # is needed. Values are signed metres, positive inside the polygon.
        return torch.einsum("...ej,ej->...e", points[..., None, :] - self.polygon, self.inward)


def _patch(record, reference):
    _provenance(record.get("source"), "Support surface")
    origin = _tensor(record.get("origin"), reference, "Surface origin", (3,))
    normal = _tensor(record.get("normal"), reference, "Surface normal", (3,))
    length = normal.norm()
    if length.item() < 1e-8:
        raise ValueError("Surface normal must be nonzero and point into free space")
    normal = normal / length
    polygon = _tensor(record.get("polygon_world"), reference, "Surface polygon")
    if polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3:
        raise ValueError("Surface polygon_world requires at least three ordered world vertices")
    if ((polygon - origin) @ normal).abs().max().item() > 1e-5:
        raise ValueError("Surface polygon must be coplanar with its origin and normal")
    edges = polygon.roll(-1, dims=0) - polygon
    if edges.norm(dim=-1).min().item() < 1e-7:
        raise ValueError("Surface polygon has a duplicate adjacent vertex")
    area = (torch.cross(polygon - origin, polygon.roll(-1, dims=0) - origin, dim=-1) @ normal).sum()
    if abs(area.item()) < 1e-10:
        raise ValueError("Surface polygon has zero signed area")
    inward = torch.cross(normal.expand_as(edges), edges, dim=-1) * area.sign()
    inward = inward / inward.norm(dim=-1, keepdim=True)
    patch = _Patch(origin, normal, polygon, inward)
    # Every vertex must lie in every inward half-space; this rejects concave
    # and unordered/self-crossing polygons instead of taking their convex hull.
    if patch.edge_distances(polygon).min().item() < -1e-6:
        raise ValueError("Surface polygon must be ordered and convex")
    return patch


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int
    patch: _Patch
    ids: torch.Tensor
    weights: torch.Tensor
    height_weights: torch.Tensor
    tangent_weights: torch.Tensor
    anchor: torch.Tensor | None
    boundary_margin: float


class SceneSupportConstraint:
    """Compile a ``grail.scene_support.v1`` manifest once per native fit.

    ``regions`` comes from ``foot_region_indices(human_model)`` and therefore
    preserves native SOMA topology and independent heel/toe roles.
    The manifest requires frame_count, source, surfaces and intervals. Surface
    records have id, origin, normal (into free space), polygon_world and source.
    Intervals have part (FOOT_PARTS name), start inclusive, end exclusive,
    surface_id and source; overlapping intervals for the same part are rejected.

    Optional interval controls: weight, blend_frames, contact_vertex_ids,
    anchor_world and boundary_margin_m. A blend attenuates only frames inside
    the interval and keeps its first/last frame active; it does not extend
    attraction into swing. Fixed contact_vertex_ids select a native sole patch;
    omit only when a full-part centroid proxy is appropriate. All-region
    vertices still participate in the all-frame penetration term.
    Optional constructor height_blend_frames/tangent_blend_frames override only
    that term's ramp; None preserves the manifest exactly. Boundary containment
    retains the manifest ramp. Each term normalizes its own active weights.

    Without anchor_world, each interval's tangent anchor is its differentiable
    weighted mean, analytically eliminating a free shared anchor. This penalizes
    drift while allowing the entire stance to move together; separate stances
    do not pull on one another. Boundary containment constrains the contact
    patch centroid, not the whole foot volume or a support polygon of the body.

    penetration_mode defaults to normal_depth, preserving the original finite
    support-patch penalty. Explicit column_exit requires collision_surfaces
    with the same schema as surfaces. They describe the original full solid
    footprints, including the floor, independently of exposed support patches.
    The column proxy is min(depth below top, inward side distance), zero outside
    or above the column. It avoids a jump at the footprint boundary and gives
    a tangent escape gradient near a side. Overlapping original columns avoid
    artificial interior seams introduced by splitting exposed support treads.
    """

    def __init__(
        self,
        vertices,
        regions,
        manifest,
        *,
        robust_delta_m=0.005,
        height_blend_frames=None,
        tangent_blend_frames=None,
    ):
        if (
            vertices.ndim != 3
            or vertices.shape[-1] != 3
            or not vertices.shape[0]
            or not vertices.is_floating_point()
        ):
            raise ValueError("Scene support requires native floating vertices [T,V,3], T > 0")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("Scene support initialization must be finite")
        self.delta = _nonnegative(robust_delta_m, "robust_delta_m")
        if self.delta == 0:
            raise ValueError("robust_delta_m must be positive")
        for name, value in (
            ("height_blend_frames", height_blend_frames),
            ("tangent_blend_frames", tangent_blend_frames),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a nonnegative integer")
        if len(regions) != 4:
            raise ValueError("Scene support requires four native heel/toe regions")
        native_regions = tuple(_ids(ids, vertices, part) for part, ids in zip(FOOT_PARTS, regions))
        self.ids = torch.unique(torch.cat(native_regions))
        self.shape = vertices.shape
        self.dtype = vertices.dtype
        self.device = vertices.device
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "grail.scene_support.v1":
            raise ValueError("Expected grail.scene_support.v1 manifest")
        if (
            type(manifest.get("frame_count")) is not int
            or manifest["frame_count"] != vertices.shape[0]
        ):
            raise ValueError("Scene support frame_count differs from native sequence")
        _provenance(manifest.get("source"), "Scene support manifest")
        surfaces = manifest.get("surfaces")
        if not isinstance(surfaces, list) or not surfaces:
            raise ValueError("Scene support requires nonempty surfaces")
        self.patches = {}
        for record in surfaces:
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("id"), str)
                or not record["id"]
            ):
                raise ValueError("Each support surface requires a string id")
            if record["id"] in self.patches:
                raise ValueError("Duplicate support surface id")
            self.patches[record["id"]] = _patch(record, vertices)
        self.penetration_mode = manifest.get("penetration_mode", "normal_depth")
        if self.penetration_mode not in ("normal_depth", "column_exit"):
            raise ValueError("penetration_mode must be normal_depth or column_exit")
        self.collision_patches = self.patches
        if self.penetration_mode == "column_exit":
            columns = manifest.get("collision_surfaces")
            if not isinstance(columns, list) or not columns:
                raise ValueError(
                    "column_exit requires declared original collision_surfaces, including floor"
                )
            self.collision_patches = {}
            for record in columns:
                if (
                    not isinstance(record, Mapping)
                    or not isinstance(record.get("id"), str)
                    or not record["id"]
                ):
                    raise ValueError("Each collision surface requires a string id")
                if record["id"] in self.collision_patches:
                    raise ValueError("Duplicate collision surface id")
                self.collision_patches[record["id"]] = _patch(record, vertices)
        intervals = manifest.get("intervals")
        if not isinstance(intervals, list):
            raise ValueError("Scene support requires an intervals list (may be empty)")
        occupied = {part: set() for part in FOOT_PARTS}
        self.intervals = []
        for record in intervals:
            if not isinstance(record, Mapping):
                raise ValueError("Support interval must be a mapping")
            _provenance(record.get("source"), "Support interval")
            part, start, end = record.get("part"), record.get("start"), record.get("end")
            if (
                part not in FOOT_PARTS
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= vertices.shape[0]
            ):
                raise ValueError(
                    "Support interval requires a native part and valid [start,end) frames"
                )
            if occupied[part].intersection(range(start, end)):
                raise ValueError("Overlapping support intervals for the same part")
            occupied[part].update(range(start, end))
            surface_id = record.get("surface_id")
            if not isinstance(surface_id, str) or surface_id not in self.patches:
                raise ValueError("Support interval references an unknown surface_id")
            patch = self.patches[surface_id]
            ids = native_regions[FOOT_PARTS.index(part)]
            if "contact_vertex_ids" in record:
                selected = _ids(record["contact_vertex_ids"], vertices, "Contact patch")
                if not bool(torch.isin(selected, ids).all()):
                    raise ValueError("contact_vertex_ids must be a subset of the native part")
                ids = selected
            weight = _nonnegative(record.get("weight", 1.0), "Interval weight")
            if weight == 0:
                raise ValueError("Omit inactive intervals instead of setting zero weight")
            blend = record.get("blend_frames", 0)
            if type(blend) is not int or blend < 0:
                raise ValueError("blend_frames must be a nonnegative integer")

            weights = _phase_weights(vertices, end - start, weight, blend)
            height_weights = (
                weights
                if height_blend_frames is None
                else _phase_weights(vertices, end - start, weight, height_blend_frames)
            )
            tangent_weights = (
                weights
                if tangent_blend_frames is None
                else _phase_weights(vertices, end - start, weight, tangent_blend_frames)
            )
            margin = _nonnegative(record.get("boundary_margin_m", 0.0), "Boundary margin")
            anchor = None
            if "anchor_world" in record:
                anchor = _tensor(record["anchor_world"], vertices, "Interval anchor", (3,))
                if (
                    abs(patch.heights(anchor).item()) > 1e-5
                    or patch.edge_distances(anchor).min().item() < margin - 1e-6
                ):
                    raise ValueError(
                        "anchor_world must lie on the selected patch within its boundary margin"
                    )
            self.intervals.append(
                _Interval(
                    start, end, patch, ids, weights, height_weights, tangent_weights, anchor, margin
                )
            )

    def _robust(self, value):
        magnitude = value.abs()
        return torch.where(
            magnitude < self.delta, 0.5 * value.square() / self.delta, magnitude - 0.5 * self.delta
        )

    def terms(self, vertices):
        """Return unweighted metre-Huber terms with no per-iteration host sync."""
        if (
            vertices.shape != self.shape
            or vertices.device != self.device
            or vertices.dtype != self.dtype
        ):
            raise ValueError("Scene support native layout/device/dtype changed within a fit")
        points = vertices[:, self.ids]
        penetrations = []
        for patch in self.collision_patches.values():
            edges = patch.edge_distances(points)
            below = (-patch.heights(points)).clamp_min(0)
            if self.penetration_mode == "column_exit":
                depth = torch.minimum(below, edges.amin(dim=-1).clamp_min(0))
            else:
                depth = below * (edges >= -1e-7).all(dim=-1)
            penetrations.append(depth.amax(dim=1))
        worst = torch.stack(penetrations).amax(dim=0)
        zero = vertices.sum() * 0.0
        result = {
            "penetration": self._robust(worst).mean(),
            "support_height": zero,
            "tangent_anchor": zero,
            "boundary": zero,
        }
        denominators = {
            key: vertices.new_zeros(()) for key in ("support_height", "tangent_anchor", "boundary")
        }
        for interval in self.intervals:
            native_patch = vertices[interval.start : interval.end, interval.ids]
            height = interval.patch.heights(native_patch).amin(dim=1)
            center = native_patch.mean(dim=1)
            anchor = interval.anchor
            if anchor is None:
                anchor = (center * interval.tangent_weights[:, None]).sum(
                    dim=0
                ) / interval.tangent_weights.sum()
            offset = center - anchor
            tangent = offset - (offset @ interval.patch.normal)[:, None] * interval.patch.normal
            outside = (
                (interval.boundary_margin - interval.patch.edge_distances(center))
                .clamp_min(0)
                .amax(dim=-1)
            )
            result["support_height"] = (
                result["support_height"] + (self._robust(height) * interval.height_weights).sum()
            )
            result["tangent_anchor"] = (
                result["tangent_anchor"]
                + (self._robust(tangent.norm(dim=-1)) * interval.tangent_weights).sum()
            )
            result["boundary"] = (
                result["boundary"] + (self._robust(outside) * interval.weights).sum()
            )
            denominators["support_height"] = (
                denominators["support_height"] + interval.height_weights.sum()
            )
            denominators["tangent_anchor"] = (
                denominators["tangent_anchor"] + interval.tangent_weights.sum()
            )
            denominators["boundary"] = denominators["boundary"] + interval.weights.sum()
        for key in ("support_height", "tangent_anchor", "boundary"):
            result[key] = result[key] / denominators[key].clamp_min(1e-12)
        return result
