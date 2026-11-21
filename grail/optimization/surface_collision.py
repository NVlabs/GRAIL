"""Optional area-normalized hand/object overlap in current native geometry.

The immutable union grid contains already-scaled object-local metres. It is a
3 mm trilinear approximation to the minimum closed-component signed distance,
not an exact Boolean-union escape distance. This is a kinematic collision
penalty, not a dynamics, hand-hand, or tabletop check. Existing losses and
default paths are untouched; this module is disabled unless explicitly enabled.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .visible_skin import canonical_surface_quadrature


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def canonical_hash(value, kind):
    """Hash original ordering, with explicit little-endian canonical dtype."""
    dtype = "<i8" if kind == "faces" else "<f8"
    return hashlib.sha256(np.ascontiguousarray(_array(value), dtype=dtype).tobytes()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VerifiedUnionSignedGrid(nn.Module):
    """Hash- and geometry-verified signed-only grid; query coordinates are metres."""

    def __init__(
        self, *, grid_path, expected_grid_sha256, object_vertices_scaled, object_faces, object_scale
    ):
        super().__init__()
        path = Path(grid_path)
        if (
            not isinstance(expected_grid_sha256, str)
            or len(expected_grid_sha256) != 64
            or _sha(path) != expected_grid_sha256
        ):
            raise ValueError("Union collision grid file hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            required = {
                "union_signed",
                "lower_xyz",
                "upper_xyz",
                "spacing_m",
                "axis_x",
                "axis_y",
                "axis_z",
                "object_raw_vertex_ids",
                "object_raw_vertices_scaled_m",
                "object_raw_faces",
                "metadata_json",
            }
            if not required.issubset(z.files):
                raise ValueError("Incomplete union collision grid schema")
            metadata = json.loads(str(z["metadata_json"]))
            grid, lower, upper = (
                z["union_signed"].copy(),
                z["lower_xyz"].copy(),
                z["upper_xyz"].copy(),
            )
            spacing = float(z["spacing_m"])
            raw, faces = z["object_raw_vertices_scaled_m"].copy(), z["object_raw_faces"].copy()
            scale = _array(object_scale).reshape(-1)
            if (
                scale.shape != (3,)
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
                or not np.array_equal(scale, metadata["object_scale_fixed"])
            ):
                raise ValueError("Object scale differs from the already-scaled union grid")
            supplied_v, supplied_f = _array(object_vertices_scaled), _array(object_faces)
            if (
                supplied_f.dtype.kind not in "iu"
                or not np.array_equal(supplied_v, raw)
                or not np.array_equal(supplied_f, faces)
            ):
                raise ValueError(
                    "Original canonical object geometry/topology mismatch; simplified tracking IDs are invalid"
                )
            if (
                raw.ndim != 2
                or raw.shape[1] != 3
                or not np.isfinite(raw).all()
                or faces.ndim != 2
                or faces.shape[1] != 3
                or faces.dtype.kind not in "iu"
                or not len(faces)
                or faces.min() < 0
                or faces.max() >= len(raw)
            ):
                raise ValueError("Invalid original canonical object arrays")
            if not np.array_equal(z["object_raw_vertex_ids"], np.arange(len(raw))):
                raise ValueError("Original object vertex ordering is not preserved")
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(upper <= lower)
                or not math.isfinite(spacing)
                or spacing <= 0
            ):
                raise ValueError("Invalid signed-grid bounds or spacing")
            axes = [z[k] for k in ("axis_x", "axis_y", "axis_z")]
            if grid.shape != tuple(len(a) for a in axes[::-1]) or not np.isfinite(grid).all():
                raise ValueError("Invalid z,y,x union grid")
            for i, axis in enumerate(axes):
                if (
                    len(axis) < 2
                    or axis[0] != lower[i]
                    or axis[-1] != upper[i]
                    or not np.allclose(np.diff(axis), spacing, rtol=1e-6, atol=1e-12)
                ):
                    raise ValueError("Signed-grid axes disagree with bounds/spacing")
            if np.any(raw.min(0) <= lower) or np.any(raw.max(0) >= upper):
                raise ValueError("Object must be strictly contained in the union grid")
        for name, value in [("grid", grid[None, None]), ("lower", lower), ("upper", upper)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        self.provenance = dict(
            grid_path=str(path.resolve()),
            grid_sha256=expected_grid_sha256,
            channel="union_signed",
            object_vertices_sha256=canonical_hash(raw, "vertices"),
            object_faces_sha256=canonical_hash(faces, "faces"),
            object_vertex_count=len(raw),
            object_face_count=len(faces),
            object_scale=scale.tolist(),
            spacing_m=spacing,
            cache_metadata=metadata,
            outside_grid="Nonnegative border value plus distance to bounds; no exterior penetration penalty.",
            caveat="Trilinear grid and min-component signed distance approximate overlap. Depth can underestimate Boolean-union escape through overlapping components.",
        )

    def query(self, points):
        if points.shape[-1] != 3 or not points.is_floating_point():
            raise ValueError("Signed-grid query needs floating-point scaled object-local xyz")
        if points.device != self.grid.device or points.dtype != self.grid.dtype:
            raise ValueError(
                "Move/cast collision module to the current native pose device and dtype first"
            )
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        clamped = torch.maximum(torch.minimum(flat, self.upper), self.lower)
        outside = torch.linalg.vector_norm(flat - clamped, dim=-1)
        query = (2 * (clamped - self.lower) / (self.upper - self.lower) - 1).reshape(1, 1, 1, -1, 3)
        signed = F.grid_sample(
            self.grid, query, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0, 0, 0]
        return torch.where(outside > 0, signed.clamp_min(0) + outside, signed).reshape(shape)


class _HandSamples(nn.Module):
    def __init__(self, ids, bary, area):
        super().__init__()
        self.register_buffer("original_face_ids", torch.as_tensor(ids, dtype=torch.long))
        self.register_buffer("barycentric", torch.as_tensor(bary, dtype=torch.float64))
        self.register_buffer("area_m2", torch.as_tensor(area, dtype=torch.float64))


class SurfaceCollisionLoss(nn.Module):
    """Mean of per-hand area means, then frames, of squared excess depth (m²).

    ``rest_vertices`` are the original native human template with scale baked.
    ``hand_face_ids`` explicitly supplies complete original left/right hand
    regions. Topology/rest hashes pin their meaning; no closest-hand labeling
    or simplified object indexing is performed. The user of this module owns
    anatomical region review and independent whole-body collision evaluation.

    For each hand and frame: sum(area * relu(-SDF - tolerance)²) / sum(area).
    The denominator includes all fixed canonical samples. Source confidence,
    visibility, and current penetration fraction do not reweight collision.
    Both hands get equal weight regardless of area; no top-k/tail term is added.
    Canonical sample edges are <=2 mm; skinning may stretch posed edges.
    """

    def __init__(
        self,
        *,
        human_faces,
        rest_vertices,
        hand_face_ids,
        expected_human_faces_sha256,
        expected_human_rest_sha256,
        grid_path,
        expected_grid_sha256,
        object_vertices_scaled,
        object_faces,
        object_scale,
        enabled=False,
        tolerance_m=0.0005,
        max_edge_m=0.002,
    ):
        super().__init__()
        if not isinstance(enabled, bool) or not math.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("Explicit boolean enabled and finite nonnegative tolerance required")
        faces, rest = _array(human_faces), _array(rest_vertices)
        if (
            faces.dtype.kind not in "iu"
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or rest.ndim != 2
            or rest.shape[1] != 3
            or not np.isfinite(rest).all()
            or faces.min() < 0
            or faces.max() >= len(rest)
        ):
            raise ValueError("Invalid original human topology/template")
        if (
            canonical_hash(faces, "faces") != expected_human_faces_sha256
            or canonical_hash(rest, "vertices") != expected_human_rest_sha256
        ):
            raise ValueError("Original human topology or baked template hash mismatch")
        if set(hand_face_ids) != {"left", "right"}:
            raise ValueError("Explicit original left and right hand face regions required")
        if np.intersect1d(_array(hand_face_ids["left"]), _array(hand_face_ids["right"])).size:
            raise ValueError("Left/right original face regions overlap")
        self.field = VerifiedUnionSignedGrid(
            grid_path=grid_path,
            expected_grid_sha256=expected_grid_sha256,
            object_vertices_scaled=object_vertices_scaled,
            object_faces=object_faces,
            object_scale=object_scale,
        )
        self.register_buffer("human_faces", torch.as_tensor(faces, dtype=torch.long))
        self.human_vertex_count = len(rest)
        self.enabled, self.tolerance_m = enabled, float(tolerance_m)
        self.samples = nn.ModuleDict()
        regions = {}
        for side in ("left", "right"):
            original_ids = _array(hand_face_ids[side])
            if original_ids.dtype.kind not in "iu":
                raise ValueError("Hand faces must be integer original face IDs")
            ids, bary, area, info = canonical_surface_quadrature(
                rest, faces, original_ids, max_edge_m
            )
            self.samples[side] = _HandSamples(ids, bary, area)
            info["area_rule"] = (
                "Fixed canonical represented area once; denominator includes all hand samples in every frame."
            )
            regions[side] = dict(
                **info,
                original_face_ids=original_ids.tolist(),
                sampled_face_ids_sha256=canonical_hash(ids, "faces"),
                barycentric_sha256=canonical_hash(bary, "vertices"),
                area_sha256=canonical_hash(area, "vertices"),
            )
        self.provenance = dict(
            object_field=self.field.provenance,
            human_faces_sha256=expected_human_faces_sha256,
            human_rest_sha256=expected_human_rest_sha256,
            human_vertex_count=len(rest),
            human_face_count=len(faces),
            hand_regions=regions,
            tolerance_m=self.tolerance_m,
            units="m^2",
            normalization="Equal per-hand area means, then equal frame mean.",
            source_confidence="Not used; geometric collision only.",
            physical_simulation=False,
            scene_camera_dependency=False,
            sampling_function="grail.optimization.visible_skin.canonical_surface_quadrature",
            scope="Supplied original left/right hand surfaces only; no hand-hand, table, forearm or whole-human collision guarantee.",
        )
        self.last_details = {}

    def forward(self, human_vertices, object_R, object_t):
        """Live full native human mesh and matching current object rigid poses."""
        if (
            human_vertices.ndim != 3
            or human_vertices.shape[1:] != (self.human_vertex_count, 3)
            or len(human_vertices) == 0
        ):
            raise ValueError(
                "Supply the current full original native human mesh [frames,vertices,3]"
            )
        frames = len(human_vertices)
        if object_R.shape != (frames, 3, 3) or object_t.shape != (frames, 3):
            raise ValueError("Current object rigid transforms must match the human frames")
        for value in (human_vertices, object_R, object_t):
            if (
                value.device != self.field.grid.device
                or value.dtype != self.field.grid.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "Current poses and module must share finite floating-point device/dtype"
                )
        with torch.no_grad():
            identity = torch.eye(3, device=object_R.device, dtype=object_R.dtype)
            if not bool(
                torch.allclose(
                    object_R.transpose(-1, -2) @ object_R,
                    identity.expand(frames, -1, -1),
                    atol=1e-4,
                    rtol=1e-4,
                )
            ) or bool((torch.linalg.det(object_R) <= 0).any()):
                raise ValueError(
                    "Object transforms must be rigid rotations; no scale or reflection"
                )
        zero = human_vertices.sum() * 0 + object_R.sum() * 0 + object_t.sum() * 0
        if not self.enabled:
            self.last_details = dict(enabled=False, frames=frames, units="m^2")
            return dict(
                total=zero,
                per_hand_m2={"left": zero, "right": zero},
                per_hand_frame_m2=zero.expand(frames, 2),
            )
        per_hand = []
        details = {}
        for side in ("left", "right"):
            sample = self.samples[side]
            triangles = human_vertices[:, self.human_faces[sample.original_face_ids]]
            world = (triangles * sample.barycentric[None, :, :, None]).sum(-2)
            local = (world - object_t[:, None]) @ object_R
            signed = self.field.query(local)
            excess = F.relu(-signed - self.tolerance_m)
            area = sample.area_m2
            loss = (excess.square() * area).sum(-1) / area.sum()
            per_hand.append(loss)
            with torch.no_grad():
                details[side] = dict(
                    samples=len(area),
                    represented_area_m2=float(area.sum()),
                    excess_depth_max_m=excess.max(-1).values.cpu().tolist(),
                    signed_min_m=signed.min(-1).values.cpu().tolist(),
                    penetrating_area_fraction=((signed < 0) * area)
                    .sum(-1)
                    .div(area.sum())
                    .cpu()
                    .tolist(),
                    beyond_tolerance_area_fraction=((excess > 0) * area)
                    .sum(-1)
                    .div(area.sum())
                    .cpu()
                    .tolist(),
                )
        values = torch.stack(per_hand, -1)
        self.last_details = dict(enabled=True, frames=frames, units="m^2", hands=details)
        return dict(
            total=values.mean(),
            per_hand_m2={side: values[:, i].mean() for i, side in enumerate(("left", "right"))},
            per_hand_frame_m2=values,
        )
