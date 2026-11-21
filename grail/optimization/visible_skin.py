"""Optional sparse RGB skin coverage with current object/full-human occlusion.

Source observations are visible skin pixels, never hidden contact-pad IDs.
Candidate regions retain explicit original human face IDs. Closest projected
triangle points use perspective-correct barycentrics; image and occlusion costs
are charged on the SAME live surface point before latent matching. Warp selects
current first-hit faces and Torch differentiates their ray-plane depths. Face
switches and closest-feature switches are piecewise constant.
"""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from torch import nn


@wp.kernel
def _first_hit(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[i], directions[i], 100000.0)
    faces[i] = -1
    if q.result:
        faces[i] = q.face


def closest_projected_triangle(observed, triangles):
    """Exact closest filled-triangle screen point, including degenerate edges."""
    a, b, c = triangles.unbind(-2)
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    den = cross(b - a, c - a)
    good = den.abs() > 1e-12
    safe = torch.where(good, den, torch.ones_like(den))
    wb = cross(observed - a, c - a) / safe
    wc = cross(b - a, observed - a) / safe
    inside_bary = torch.stack((1 - wb - wc, wb, wc), -1)
    inside = (good & (inside_bary >= 0).all(-1)).detach()
    starts = torch.stack((a, b, c), -2)
    ends = torch.stack((b, c, a), -2)
    edge = ends - starts
    t = (
        ((observed[..., None, :] - starts) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-12)
    ).clamp(0, 1)
    candidates = starts + t[..., None] * edge
    selected = (candidates - observed[..., None, :]).square().sum(-1).detach().argmin(-1)
    t = t.gather(-1, selected[..., None]).squeeze(-1)
    edge_bary = (1 - t[..., None]) * F.one_hot(selected, 3) + t[..., None] * F.one_hot(
        (selected + 1) % 3, 3
    )
    bary = torch.where(inside[..., None], inside_bary, edge_bary)
    return (bary[..., None] * triangles).sum(-2), bary


def ray_depth_excess(points, origin, triangles, hit, tolerance):
    ray = points - origin
    distance = ray.norm(dim=-1)
    direction = ray / distance[..., None].clamp_min(1e-12)
    a, b, c = triangles.unbind(-2)
    normal = F.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1)
    den = (normal * direction).sum(-1)
    valid = hit & (den.abs() > 1e-7)
    depth = ((a - origin) * normal).sum(-1) / torch.where(valid, den, torch.ones_like(den))
    valid = valid & (depth > 0)
    return torch.where(valid, F.relu(distance - depth - tolerance), torch.zeros_like(distance))


def canonical_surface_quadrature(rest_vertices, human_faces, face_ids, max_edge_m=0.002):
    """Area-conserving nodes with original face IDs and canonical barycentrics.

    Subdivide each original triangle uniformly into n^2 triangles, where n is
    ceil(longest canonical edge / max_edge_m). Each child triangle contributes
    area/3 at each node. Seam nodes stay separate by original face; their masses
    partition incident face area without duplicating it. The edge bound applies
    to canonical geometry; skinning may stretch current edges.
    """
    if not np.isfinite(max_edge_m) or not 0 < max_edge_m <= 0.002:
        raise ValueError("Visible surface canonical sampling requires 0 < max_edge_m <= 0.002")
    rest = np.asarray(rest_vertices, dtype=np.float64)
    faces = np.asarray(human_faces, dtype=np.int64)
    ids = np.asarray(face_ids, dtype=np.int64)
    if (
        ids.ndim != 1
        or not len(ids)
        or len(np.unique(ids)) != len(ids)
        or ids.min() < 0
        or ids.max() >= len(faces)
    ):
        raise ValueError("Expected unique original human face IDs for quadrature")
    owners = []
    barycentrics = []
    masses = []
    max_actual_edge = 0.0
    subtriangle_count = 0
    for face_id in ids:
        tri = rest[faces[face_id]]
        edge = np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=-1).max()
        area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2
        if not np.isfinite(tri).all() or not np.isfinite(area) or area <= 0:
            raise ValueError("Degenerate or nonfinite canonical visible surface triangle")
        n = max(1, int(np.ceil(edge / max_edge_m)))
        ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        index = {p: k for k, p in enumerate(ij)}
        bary = np.array([[1 - (i + j) / n, i / n, j / n] for i, j in ij])
        children = []
        for i in range(n):
            for j in range(n - i):
                children.append([index[i, j], index[i + 1, j], index[i, j + 1]])
                if i + j <= n - 2:
                    children.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
        children = np.asarray(children, dtype=np.int64)
        nodal = np.bincount(children.ravel(), minlength=len(bary)) * area / (3 * n * n)
        if len(children) != n * n or not np.isclose(nodal.sum(), area, rtol=1e-12, atol=1e-16):
            raise ValueError("Canonical visible surface quadrature failed area conservation")
        owners.extend([int(face_id)] * len(bary))
        barycentrics.extend(bary)
        masses.extend(nodal)
        max_actual_edge = max(max_actual_edge, float(edge / n))
        subtriangle_count += n * n
    return (
        np.asarray(owners, dtype=np.int64),
        np.asarray(barycentrics),
        np.asarray(masses),
        {
            "original_faces": len(ids),
            "samples": len(owners),
            "auxiliary_triangles": subtriangle_count,
            "max_canonical_edge_m": max_actual_edge,
            "represented_area_m2": float(np.sum(masses)),
            "area_rule": "Retained raw canonical nodal area normalized once after current visibility selection; source confidence separate.",
            "canonical_edge_bound_is_not_posed_edge_bound": True,
        },
    )


class _VisibleSurfaceRegion(nn.Module):
    def __init__(self, face_ids, barycentric, area):
        super().__init__()
        self.register_buffer("face_ids", torch.as_tensor(face_ids, dtype=torch.long))
        self.register_buffer("barycentric", torch.as_tensor(barycentric, dtype=torch.float64))
        self.register_buffer("area_m2", torch.as_tensor(area, dtype=torch.float64))


class VisibleSkinLoss(nn.Module):
    """Dimensionless mean of source-confidence times latent pixel/depth cost.

    `observations`: list(frame, region, xy, uncertainty_px, confidence).
    `regions`: dictionary name -> original human triangle IDs.
    No interpolation: only explicitly annotated frames contribute. Candidate
    areas are fixed canonical quadrature priors and normalized before matching;
    source confidence is applied once outside the latent cost, without division
    by summed source confidence. Full original human mesh supplies self blockers.

    Optional ``matching_mode='visible_surface_samples'`` uses canonical <=2mm
    original-face/barycentric nodes and matches only currently visible samples.
    Human first-hit depth must agree within0.5mm and the object must lie at least
    0.5mm behind the sample (or its ray must miss). Visibility is a detached hard
    selection: this mode has no direct object-pose gradient. Object coupling
    must come from separately configured contact/tracking/RGB terms. No visible
    candidates raises a local frame/region diagnostic, not global infeasibility.
    """

    def __init__(
        self,
        *,
        human_faces,
        rest_vertices,
        object_vertices,
        object_faces,
        camera_R,
        camera_t,
        focal_length_px,
        image_size,
        observations,
        regions,
        frame_count,
        device="cuda:0",
        temperature=0.1,
        depth_scale_m=0.01,
        depth_tolerance_m=0.0005,
        pixel_guard_weight=1.0,
        occlusion_weight=1.0,
        matching_mode="closest_triangle",
        surface_max_edge_m=0.002,
        capture_support=False,
        region_objective_multipliers=None,
    ):
        super().__init__()
        if region_objective_multipliers is not None and not isinstance(
            region_objective_multipliers, dict
        ):
            raise TypeError("Region objective multipliers must be a dictionary or None")
        supplied = {} if region_objective_multipliers is None else region_objective_multipliers
        if set(supplied) - set(regions):
            raise ValueError("Unknown region objective multiplier key")
        for name, value in supplied.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Region objective multipliers must be finite nonnegative numbers")
        self.region_objective_multipliers = {
            name: float(supplied.get(name, 1.0)) for name in regions
        }
        self._region_multipliers_explicit = region_objective_multipliers is not None
        wp.init()
        if (
            min(temperature, depth_scale_m, frame_count) <= 0
            or min(depth_tolerance_m, pixel_guard_weight, occlusion_weight) < 0
        ):
            raise ValueError("Invalid visible-skin settings")
        self.temperature = temperature
        self.depth_scale_m = depth_scale_m
        self.depth_tolerance_m = depth_tolerance_m
        self.pixel_guard_weight = pixel_guard_weight
        self.occlusion_weight = occlusion_weight
        self.frame_count = frame_count
        self.observations = observations
        self.regions = {}
        self._human_meshes = {}
        self._object_mesh = None
        if matching_mode not in ("closest_triangle", "visible_surface_samples"):
            raise ValueError("Unknown visible-skin matching mode")
        self.matching_mode = matching_mode
        if type(capture_support) is not bool:
            raise TypeError("capture_support must be boolean")
        if capture_support and matching_mode != "visible_surface_samples":
            raise ValueError("Support capture requires strict visible-surface mode")
        self.capture_support = capture_support
        self.last_support = None
        self._support_pool_hashes = {}
        self.surface_visibility_margin_m = 0.0005
        self.surface_regions = nn.ModuleList()
        self.surface_region_indices = {}
        self.surface_quadrature_provenance = {}
        self.human_vertex_count = len(rest_vertices)
        hf = np.asarray(human_faces, np.int64)
        for name, ids in regions.items():
            ids = np.asarray(ids, np.int64)
            if (
                ids.ndim != 1
                or not len(ids)
                or ids.min() < 0
                or ids.max() >= len(hf)
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("Invalid original region triangle IDs")
            triangles = np.asarray(rest_vertices)[hf[ids]]
            area = (
                np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=-1,
                )
                / 2
            )
            if (area <= 0).any():
                raise ValueError("Degenerate canonical candidate triangle")
            self.regions[name] = (
                torch.as_tensor(ids, device=device),
                torch.as_tensor(area / area.sum(), device=device, dtype=torch.float32),
            )
            if matching_mode == "visible_surface_samples":
                owner, bary, mass, provenance = canonical_surface_quadrature(
                    rest_vertices, hf, ids, surface_max_edge_m
                )
                self.surface_region_indices[name] = len(self.surface_regions)
                self.surface_regions.append(_VisibleSurfaceRegion(owner, bary, mass).to(device))
                self.surface_quadrature_provenance[name] = provenance
        for a in observations:
            if (
                a["region"] not in regions
                or not 0 <= a["frame"] < frame_count
                or a["uncertainty_px"] <= 0
                or not 0 < a["confidence"] <= 1
                or np.asarray(a["xy"]).shape != (2,)
                or not np.isfinite(a["xy"]).all()
            ):
                raise ValueError("Invalid source skin observation")
        if not observations:
            raise ValueError("No source skin observations")
        for name, value in [
            ("human_faces", hf),
            ("object_faces", object_faces),
            ("object_local", object_vertices),
            ("camera_R", camera_R),
            ("camera_t", camera_t),
            ("image_center", np.asarray(image_size) / 2),
            ("focal", focal_length_px),
        ]:
            self.register_buffer(
                name,
                torch.as_tensor(
                    value,
                    device=device,
                    dtype=torch.long if name.endswith("faces") else torch.float32,
                ),
            )
        self.device_name = str(torch.device(device))
        self.last_details = []

    def project(self, points):
        camera = (points - self.camera_t) @ self.camera_R
        if (camera[..., 2] >= -1e-6).any():
            raise ValueError("Skin triangles cross calibrated camera plane")
        return (
            camera[..., :2] * camera.new_tensor([1.0, -1.0]) / -camera[..., 2:3] * self.focal
            + self.image_center,
            -camera[..., 2],
        )

    def _hits(self, points, human, rotation, translation, frame):
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(points.device))
            if points.device.type == "cuda"
            else None
        )
        context = wp.ScopedStream(stream) if stream else wp.ScopedDevice("cpu")
        with torch.no_grad(), context:
            direction = (
                F.normalize(points - self.camera_t, dim=-1).reshape(-1, 3).float().contiguous()
            )
            if self._object_mesh is None:
                v = (self.object_local * 1000).float().contiguous()
                f = self.object_faces.flatten().int().contiguous()
                self._object_mesh = wp.Mesh(
                    points=wp.clone(wp.from_torch(v, dtype=wp.vec3)),
                    indices=wp.clone(wp.from_torch(f, dtype=wp.int32)),
                )
            current = ((human - self.camera_t) * 1000).float().contiguous()
            indices = self.human_faces.flatten().int().contiguous()
            if frame not in self._human_meshes:
                self._human_meshes[frame] = wp.Mesh(
                    points=wp.clone(wp.from_torch(current, dtype=wp.vec3)),
                    indices=wp.clone(wp.from_torch(indices, dtype=wp.int32)),
                )
            mesh = self._human_meshes[frame]
            wp.copy(mesh.points, wp.from_torch(current, dtype=wp.vec3))
            mesh.refit()
            results = []
            origin = (
                (((self.camera_t - translation) @ rotation) * 1000)
                .expand(len(direction), 3)
                .float()
                .contiguous()
            )
            for m, o, d in [
                (self._object_mesh, origin, (direction @ rotation.float()).contiguous()),
                (mesh, torch.zeros_like(direction), direction),
            ]:
                faces = torch.empty(len(direction), device=points.device, dtype=torch.int32)
                wp.launch(
                    _first_hit,
                    len(direction),
                    inputs=[
                        m.id,
                        wp.from_torch(o, dtype=wp.vec3),
                        wp.from_torch(d, dtype=wp.vec3),
                        wp.from_torch(faces, dtype=wp.int32),
                    ],
                    device=self.device_name,
                )
                results.append(faces.long().reshape(points.shape[:-1]))
        return results

    def forward(self, human_vertices, object_R, object_t, frame_indices=None):
        if human_vertices.ndim != 3 or human_vertices.shape[1:] != (self.human_vertex_count, 3):
            raise ValueError("Supply current full ORIGINAL human mesh")
        frames = (
            list(range(len(human_vertices)))
            if frame_indices is None
            else list(map(int, frame_indices))
        )
        if (
            len(frames) != len(human_vertices)
            or len(set(frames)) != len(frames)
            or any(f < 0 or f >= self.frame_count for f in frames)
        ):
            raise ValueError("Invalid source frame indices")
        if object_R.shape != (len(frames), 3, 3) or object_t.shape != (len(frames), 3):
            raise ValueError("Object pose batch mismatch")
        if self.matching_mode == "visible_surface_samples":
            return self._visible_surface_forward(human_vertices, object_R, object_t, frames)
        rows = {f: i for i, f in enumerate(frames)}
        values = []
        details = []
        for frame in frames:
            row = rows[frame]
            for region, (face_ids, area) in self.regions.items():
                observations = [
                    a for a in self.observations if a["frame"] == frame and a["region"] == region
                ]
                if not observations:
                    continue
                multiplier = self.region_objective_multipliers[region]
                if multiplier == 0:
                    details.extend(
                        {
                            "id": a.get("id"),
                            "frame": frame,
                            "region": region,
                            "status": "objective_disabled",
                            "selection_rule": "no candidate query",
                            "source_confidence": a["confidence"],
                            "region_objective_multiplier": 0.0,
                            "confidence_weighted_latent_loss": None,
                            "weighted_loss": 0.0,
                        }
                        for a in observations
                    )
                    continue
                triangles = human_vertices[row, self.human_faces[face_ids]]
                projected, depth = self.project(triangles)
                observed = human_vertices.new_tensor([a["xy"] for a in observations])
                uv, beta = closest_projected_triangle(observed[:, None], projected[None])
                alpha = beta / depth[None]
                alpha = alpha / alpha.sum(-1, keepdim=True)
                points = (alpha[..., None] * triangles[None]).sum(-2)
                object_hit, human_hit = self._hits(
                    points, human_vertices[row], object_R[row], object_t[row], frame
                )
                object_tri = (
                    self.object_local[self.object_faces[object_hit.clamp_min(0)]] @ object_R[row].T
                ) + object_t[row]
                human_tri = human_vertices[row, self.human_faces[human_hit.clamp_min(0)]]
                do = ray_depth_excess(
                    points, self.camera_t, object_tri, object_hit >= 0, self.depth_tolerance_m
                )
                dh = ray_depth_excess(
                    points, self.camera_t, human_tri, human_hit >= 0, self.depth_tolerance_m
                )
                pixel = (uv - observed[:, None]).norm(dim=-1)
                uncertainty = human_vertices.new_tensor(
                    [a["uncertainty_px"] for a in observations]
                )[:, None]
                huber = lambda x: torch.where(x < 1, 0.5 * x.square(), x - 0.5)
                cost = (
                    huber(pixel / uncertainty)
                    + self.pixel_guard_weight * F.relu(pixel / uncertainty - 1).square()
                    + self.occlusion_weight
                    * (huber(do / self.depth_scale_m) + huber(dh / self.depth_scale_m))
                )
                latent = -self.temperature * torch.logsumexp(
                    area.to(cost).log()[None] - cost / self.temperature, dim=-1
                )
                confidence = human_vertices.new_tensor([a["confidence"] for a in observations])
                weighted = confidence * latent
                if multiplier != 1.0:
                    weighted = weighted * multiplier
                values.extend(weighted.unbind())
                with torch.no_grad():
                    chosen = cost.argmin(-1)
                    for i, a in enumerate(observations):
                        k = chosen[i]
                        details.append(
                            {
                                "id": a.get("id"),
                                "frame": frame,
                                "region": region,
                                "selection_rule": "minimum combined cost diagnostic; excludes area prior; not latent posterior correspondence",
                                "pixel_error_px": float(pixel[i, k]),
                                "object_occlusion_m": float(do[i, k]),
                                "human_occlusion_m": float(dh[i, k]),
                                "current_human_face_id": int(face_ids[k]),
                                "current_barycentric": alpha[i, k].tolist(),
                                "weighted_loss": float(confidence[i] * latent[i]),
                            }
                        )
                        if self._region_multipliers_explicit:
                            details[-1].update(
                                source_confidence=a["confidence"],
                                region_objective_multiplier=multiplier,
                                confidence_weighted_latent_loss=details[-1]["weighted_loss"],
                                weighted_loss=float(weighted[i]),
                            )
        self.last_details = details
        zero = human_vertices.sum() * 0 + object_R.sum() * 0 + object_t.sum() * 0
        return torch.stack(values).sum() / len(self.observations) + zero if values else zero

    def _current_surface_region(self, human_vertices, object_R, object_t, row, frame, index):
        """Shared strict selection; loss and optional observer consume this same mask."""
        samples = self.surface_regions[index]
        tri = human_vertices[row, self.human_faces[samples.face_ids]]
        points = (tri * samples.barycentric.to(tri)[..., None]).sum(-2)
        projected, _ = self.project(points)
        # Piecewise-constant current visibility, never a fixed-world
        # attraction target. Invalid hit planes are not treated as misses.
        with torch.no_grad():
            object_hit, human_hit = self._hits(
                points, human_vertices[row], object_R[row], object_t[row], frame
            )
            direction = F.normalize(points - self.camera_t, dim=-1)
            distance = (points - self.camera_t).norm(dim=-1)

            def depths(vertices, faces, hit):
                triangle = vertices[faces[hit.clamp_min(0)]]
                normal = F.normalize(
                    torch.cross(
                        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=-1
                    ),
                    dim=-1,
                )
                den = (normal * direction).sum(-1)
                valid = (hit >= 0) & (den.abs() > 1e-7)
                depth = ((triangle[:, 0] - self.camera_t) * normal).sum(-1) / torch.where(
                    valid, den, torch.ones_like(den)
                )
                return depth, valid & (depth > 0)

            current_object = self.object_local @ object_R[row].T + object_t[row]
            od, ov = depths(current_object, self.object_faces, object_hit)
            hd, hv = depths(human_vertices[row], self.human_faces, human_hit)
            self_visible = hv & ((distance - hd).abs() <= self.surface_visibility_margin_m)
            object_clear = (object_hit < 0) | (
                ov & ((od - distance) > self.surface_visibility_margin_m)
            )
            visible = (self_visible & object_clear).detach()
            selected = torch.nonzero(visible).flatten()
        return dict(
            samples=samples,
            points=points,
            projected=projected,
            object_hit=object_hit,
            human_hit=human_hit,
            distance=distance,
            od=od,
            ov=ov,
            hd=hd,
            hv=hv,
            self_visible=self_visible,
            object_clear=object_clear,
            visible=visible,
            selected=selected,
        )

    def _support_identity(self, region, samples):
        if region not in self._support_pool_hashes:
            h = hashlib.sha256()
            for tensor in (
                self.human_faces,
                samples.face_ids,
                samples.barycentric,
                samples.area_m2,
            ):
                a = np.ascontiguousarray(tensor.detach().cpu().numpy())
                h.update(str(a.shape).encode())
                h.update(str(a.dtype).encode())
                h.update(a.tobytes())
            self._support_pool_hashes[region] = h.hexdigest()
        return self._support_pool_hashes[region]

    @torch.no_grad()
    def _support_record(self, frame, region, observations, current, include_membership):
        from grail.optimization.visible_support_guard import VisibleRegionSupport

        samples = current["samples"]
        selected = current["selected"]
        projected = current["projected"]
        minima = []
        for obs in observations:
            pixel = (projected[selected] - projected.new_tensor(obs["xy"])).norm(dim=-1)
            if len(selected):
                local = int(pixel.argmin())
                k = int(selected[local])
                minimum = float(pixel[local])
            else:
                k = None
                minimum = None
            minima.append(
                dict(
                    id=obs.get("id"),
                    xy=list(obs["xy"]),
                    uncertainty_px=obs["uncertainty_px"],
                    confidence=obs["confidence"],
                    minimum_visible_pixel_distance_px=minimum,
                    minimum_visible_excess_px=(
                        None if minimum is None else max(0.0, minimum - obs["uncertainty_px"])
                    ),
                    minimum_original_sample_index=k,
                    minimum_original_face_id=None if k is None else int(samples.face_ids[k]),
                )
            )
        membership = None
        if include_membership:
            # Invalid hit planes retain explicit validity flags; their numeric margin
            # placeholders are zero and must never be interpreted as clear surfaces.
            membership = {
                "sample_ids": torch.arange(len(current["points"]), device=selected.device),
                "original_face_ids": samples.face_ids,
                "barycentric": samples.barycentric,
                "canonical_area_m2": samples.area_m2,
                "visible": current["visible"],
                "self_visible": current["self_visible"],
                "object_clear": current["object_clear"],
                "human_hit_face_ids": current["human_hit"],
                "object_hit_face_ids": current["object_hit"],
                "human_depth_valid": current["hv"],
                "object_depth_valid": current["ov"],
                "object_ray_miss": current["object_hit"] < 0,
                "human_depth_difference_m": torch.where(
                    current["hv"], current["distance"] - current["hd"], 0.0
                ),
                "object_behind_margin_m": torch.where(
                    current["ov"], current["od"] - current["distance"], 0.0
                ),
            }
            membership = {k: v.detach().cpu().clone() for k, v in membership.items()}
        return VisibleRegionSupport(
            frame,
            region,
            self._support_identity(region, samples),
            "available" if len(selected) else "empty",
            float(samples.area_m2[selected].sum()),
            len(current["points"]),
            len(selected),
            int(current["self_visible"].sum()),
            int(current["object_clear"].sum()),
            tuple(minima),
            membership,
        )

    def _support_observation(self, records):
        from grail.optimization.visible_support_guard import SupportObservation

        source = hashlib.sha256(
            json.dumps(self.observations, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return SupportObservation(source, tuple(records))

    @torch.no_grad()
    def observe_support(
        self, human_vertices, object_R, object_t, frame_indices=None, *, include_membership=False
    ):
        """Observe current strict support without a scalar objective or autograd.

        Empty visibility yields typed empty region records. Invalid geometry,
        camera/input/schema errors still raise. No previous-pose mask is reused.
        Full export is explicit; capture_support=True instead publishes records
        from an ordinary strict loss's already computed mask, without extra rays.
        """
        if self.matching_mode != "visible_surface_samples":
            raise ValueError("Support observation requires strict visible-surface mode")
        if type(include_membership) is not bool:
            raise TypeError("include_membership must be boolean")
        if human_vertices.ndim != 3 or human_vertices.shape[1:] != (self.human_vertex_count, 3):
            raise ValueError("Supply current full ORIGINAL human mesh")
        frames = (
            list(range(len(human_vertices)))
            if frame_indices is None
            else list(map(int, frame_indices))
        )
        if (
            len(frames) != len(human_vertices)
            or len(set(frames)) != len(frames)
            or any(f < 0 or f >= self.frame_count for f in frames)
        ):
            raise ValueError("Invalid source frame indices")
        if object_R.shape != (len(frames), 3, 3) or object_t.shape != (len(frames), 3):
            raise ValueError("Object pose batch mismatch")
        if not all(bool(torch.isfinite(v).all()) for v in (human_vertices, object_R, object_t)):
            raise ValueError("Nonfinite support geometry or object transform")
        records = []
        for row, frame in enumerate(frames):
            for region, index in self.surface_region_indices.items():
                observations = [
                    a for a in self.observations if a["frame"] == frame and a["region"] == region
                ]
                if not observations:
                    continue
                current = self._current_surface_region(
                    human_vertices, object_R, object_t, row, frame, index
                )
                records.append(
                    self._support_record(frame, region, observations, current, include_membership)
                )
        return self._support_observation(records)

    def _visible_surface_forward(self, human_vertices, object_R, object_t, frames):
        if any(value != 1.0 for value in self.region_objective_multipliers.values()):
            raise ValueError(
                "Nonunit region objective multipliers support finite closest_triangle loss only; strict observe_support remains available"
            )
        values = []
        details = []
        self.last_details = []
        support_records = []
        if self.capture_support:
            self.last_support = None
        for row, frame in enumerate(frames):
            for region, index in self.surface_region_indices.items():
                observations = [
                    a for a in self.observations if a["frame"] == frame and a["region"] == region
                ]
                if not observations:
                    continue
                current = self._current_surface_region(
                    human_vertices, object_R, object_t, row, frame, index
                )
                samples = current["samples"]
                points = current["points"]
                projected = current["projected"]
                object_hit = current["object_hit"]
                distance = current["distance"]
                od = current["od"]
                hd = current["hd"]
                self_visible = current["self_visible"]
                object_clear = current["object_clear"]
                selected = current["selected"]
                if self.capture_support:
                    support_records.append(
                        self._support_record(frame, region, observations, current, True)
                    )
                if not len(selected):
                    failure = {
                        "frame": frame,
                        "region": region,
                        "matching_mode": self.matching_mode,
                        "status": "no_current_visible_samples",
                        "sample_count": len(points),
                        "self_visible_samples": int(self_visible.sum()),
                        "object_clear_samples": int(object_clear.sum()),
                        "scope": "Local candidate failure; does not establish global fitting infeasibility.",
                    }
                    self.last_details = details + [failure]
                    if self.capture_support:
                        self.last_support = self._support_observation(support_records)
                    raise ValueError(
                        f"No current visible human surface samples for region {region}, frame {frame}; local visibility/candidate review required"
                    )
                observed = human_vertices.new_tensor([a["xy"] for a in observations])
                pixel = (projected[selected][None] - observed[:, None]).norm(dim=-1)
                uncertainty = human_vertices.new_tensor(
                    [a["uncertainty_px"] for a in observations]
                )[:, None]
                normalized = pixel / uncertainty
                huber = torch.where(normalized < 1, 0.5 * normalized.square(), normalized - 0.5)
                cost = huber + self.pixel_guard_weight * F.relu(normalized - 1).square()
                mass = samples.area_m2[selected].to(cost)
                prior = mass / mass.sum()
                logits = prior.log()[None] - cost / self.temperature
                latent = -self.temperature * torch.logsumexp(logits, dim=-1)
                confidence = human_vertices.new_tensor([a["confidence"] for a in observations])
                values.extend((confidence * latent).unbind())
                with torch.no_grad():
                    minimum = cost.argmin(-1)
                    posterior_map = logits.argmax(-1)
                    for i, a in enumerate(observations):
                        local = minimum[i]
                        k = selected[local]
                        mk = selected[posterior_map[i]]
                        details.append(
                            {
                                "id": a.get("id"),
                                "frame": frame,
                                "region": region,
                                "matching_mode": self.matching_mode,
                                "selection_rule": "minimum pixel cost among current visible canonical samples; posterior MAP recorded separately",
                                "pixel_error_px": float(pixel[i, local]),
                                "current_human_face_id": int(samples.face_ids[k]),
                                "current_barycentric": samples.barycentric[k].tolist(),
                                "original_surface_sample_index": int(k),
                                "posterior_map_original_face_id": int(samples.face_ids[mk]),
                                "posterior_map_barycentric": samples.barycentric[mk].tolist(),
                                "visible_samples": len(selected),
                                "total_samples": len(points),
                                "retained_canonical_area_m2": float(
                                    samples.area_m2[selected].sum()
                                ),
                                "human_depth_difference_m": float(distance[k] - hd[k]),
                                "object_ray_miss": bool(object_hit[k] < 0),
                                "object_behind_margin_m": (
                                    None if object_hit[k] < 0 else float(od[k] - distance[k])
                                ),
                                "human_occlusion_m": 0.0,
                                "object_occlusion_m": 0.0,
                                "weighted_loss": float(confidence[i] * latent[i]),
                                "direct_object_gradient": False,
                            }
                        )
        self.last_details = details
        if self.capture_support:
            self.last_support = self._support_observation(support_records)
        zero = human_vertices.sum() * 0 + object_R.sum() * 0 + object_t.sum() * 0
        return torch.stack(values).sum() / len(self.observations) + zero if values else zero
