"""Optional sparse RGB object-outline fitting on live original-mesh edges.

Observations are visible object outline pixels, not contact pads or ray-inferred
underside labels. Candidate silhouette edges are recomputed from current rigid
pose and calibrated camera. Original face IDs and edge barycentrics survive
texture-seam welding (connectivity only; no vertex position is modified).
"""

import math
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grail.optimization.visible_skin import VisibleSkinLoss, ray_depth_excess


def _normal_uncertainty_weight(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Normal uncertainty weight must be a finite nonnegative number")
    return float(value)


def _normal_uncertainty_scales(observations):
    scale = []
    known = []
    for row in observations:
        sigma = row.get("normal_direction_uncertainty_degrees")
        if sigma is None:
            scale.append(1.0)
            known.append(False)
            continue
        if (
            isinstance(sigma, bool)
            or not isinstance(sigma, Real)
            or not math.isfinite(sigma)
            or not 0 < sigma <= 90
        ):
            raise ValueError("Known normal-direction uncertainty must be in(0,90]degrees")
        if "outward_image_normal_xy" not in row:
            raise ValueError("Angular uncertainty requires an observed image normal")
        denominator = 1 - math.cos(math.radians(sigma))
        if denominator <= 0:
            raise ValueError(
                "Known normal-direction uncertainty is below cosine numerical resolution"
            )
        scale.append(denominator)
        known.append(True)
    return scale, known


def add_normal_uncertainty_cost(base_cost, normal_cosine, observations, *, weight=0.0):
    """Approximate cosine cost:0.5unit at one stated angular uncertainty.

    Unknown angular uncertainty contributes zero. This is not a calibrated
    likelihood or angular marginalization. Observation confidence is applied
    once by the enclosing latent objective, never inside this candidate cost.
    """
    weight = _normal_uncertainty_weight(weight)
    if weight == 0:
        return base_cost
    if (
        base_cost.ndim != 2
        or normal_cosine.shape != base_cost.shape
        or len(observations) != len(base_cost)
    ):
        raise ValueError("Expected observation-by-candidate cost and normal cosine")
    denominator, known = _normal_uncertainty_scales(observations)
    if not any(known):
        return base_cost
    scale = base_cost.new_tensor(denominator)[:, None]
    active = torch.tensor(known, dtype=torch.bool, device=base_cost.device)[:, None]
    if not bool((scale[active] > 0).all()):
        raise ValueError("Angular uncertainty is not representable in the current cost dtype")
    penalty = 0.5 * (1 - normal_cosine.clamp(-1, 1)) / scale
    penalty = torch.where(active, penalty, torch.zeros_like(penalty))
    return base_cost + weight * penalty


def original_edge_connectivity(vertices, faces):
    """Closed manifold connectivity after EXACT coincident-position welding."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded = inverse[faces]
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    original = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    unique, group, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if not np.all(count == 2) or np.any(unique[:, 0] == unique[:, 1]):
        raise ValueError(
            "Object silhouette requires closed manifold components after exact seam welding"
        )
    ordered = np.argsort(group, kind="stable").reshape(-1, 2)
    first = ordered[:, 0]
    second = ordered[:, 1]
    edge_vertices = original[first]
    face_ids = owner[first]
    bary = np.zeros((len(first), 2, 3), np.float64)
    for k in range(2):
        bary[:, k] = faces[face_ids] == edge_vertices[:, k, None]
    if not np.all(bary.sum(-1) == 1):
        raise ValueError("Original edge barycentric provenance failed")
    return edge_vertices, np.stack([owner[first], owner[second]], axis=1), face_ids, bary


def closest_projected_segment(observed, segments):
    start, end = segments.unbind(-2)
    edge = end - start
    alpha = (((observed - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-12)).clamp(
        0, 1
    )
    return start + alpha[..., None] * edge, torch.stack([1 - alpha, alpha], -1)


class VisibleObjectBoundaryLoss(VisibleSkinLoss):
    """Reuse current ray/projection machinery; fit object silhouette curves.

    Regions contain ORIGINAL object face IDs, not tracking-mesh indices. Both
    adjacent faces must belong to the permitted region. Camera-facing sign
    changes select silhouette edges; overlapping components and full human skin
    can still block an edge and pay current ray-depth costs. Each candidate point
    pays both pixel and visibility cost before latent matching. Confidence enters
    once; a canonical edge-length prior normalizes this one-dimensional curve
    objective. This is separate from surface-area-normalized contact matching.
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
        require_visible_candidates=True,
        min_boundary_normal_cosine=0.5,
        require_exterior_candidates=False,
        normal_uncertainty_weight=0.0,
    ):
        self.normal_uncertainty_weight = _normal_uncertainty_weight(normal_uncertainty_weight)
        if self.normal_uncertainty_weight:
            _normal_uncertainty_scales(observations)
        if type(require_exterior_candidates) is not bool:
            raise TypeError("require_exterior_candidates must be boolean")
        if require_exterior_candidates and not require_visible_candidates:
            raise ValueError("Exterior eligibility requires current candidate visibility")
        # Parent initialization supplies calibrated projection and live full-mesh
        # ray blockers only. Its human candidate regions are never used below.
        faces = np.asarray(human_faces, dtype=np.int64)
        rest = np.asarray(rest_vertices)
        tri = rest[faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
        nondegenerate = np.flatnonzero(area > 0)
        if not len(nondegenerate):
            raise ValueError("No nondegenerate original human blocker mesh")
        super().__init__(
            human_faces=faces,
            rest_vertices=rest,
            object_vertices=object_vertices,
            object_faces=object_faces,
            camera_R=camera_R,
            camera_t=camera_t,
            focal_length_px=focal_length_px,
            image_size=image_size,
            observations=observations,
            regions={name: [int(nondegenerate[0])] for name in regions},
            frame_count=frame_count,
            device=device,
            temperature=temperature,
            depth_scale_m=depth_scale_m,
            depth_tolerance_m=depth_tolerance_m,
            pixel_guard_weight=pixel_guard_weight,
            occlusion_weight=occlusion_weight,
        )
        self.regions = {}
        self.require_visible_candidates = bool(require_visible_candidates)
        self.require_exterior_candidates = require_exterior_candidates
        if require_exterior_candidates:
            from grail.rendering.object_exterior import CurrentObjectExterior

            self._exterior_builder = CurrentObjectExterior()
            self.last_exterior_diagnostics = []
        if not -1 <= min_boundary_normal_cosine <= 1:
            raise ValueError("Boundary normal cosine must be in[-1,1]")
        self.min_boundary_normal_cosine = float(min_boundary_normal_cosine)
        ov = np.asarray(object_vertices)
        of = np.asarray(object_faces, dtype=np.int64)
        edge, adj, owner, bary = original_edge_connectivity(ov, of)
        tri = ov[of]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normal, axis=-1)
        if np.any(length <= 0):
            raise ValueError("Degenerate original object triangle")
        for name, value in [
            ("edge_vertices", edge),
            ("edge_adjacent_faces", adj),
            ("edge_owner_face", owner),
            ("edge_barycentric", bary),
            ("face_center", tri.mean(1)),
            ("face_normal", normal / length[:, None]),
            ("edge_length", np.linalg.norm(ov[edge[:, 1]] - ov[edge[:, 0]], axis=-1)),
        ]:
            self.register_buffer(
                name,
                torch.as_tensor(
                    value,
                    device=device,
                    dtype=(
                        torch.long
                        if name in ["edge_vertices", "edge_adjacent_faces", "edge_owner_face"]
                        else torch.float32
                    ),
                ),
            )
        for name, ids in regions.items():
            ids = np.asarray(ids, dtype=np.int64)
            if (
                ids.ndim != 1
                or not len(ids)
                or ids.min() < 0
                or ids.max() >= len(of)
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("Invalid original object-region face IDs")
            allowed = np.isin(adj, ids).all(1)
            if not allowed.any():
                raise ValueError("Object region contains no full original edges")
            self.regions[name] = torch.as_tensor(allowed, device=device)
        for observation in observations:
            if "outward_image_normal_xy" in observation:
                n = np.asarray(observation["outward_image_normal_xy"])
                if (
                    n.shape != (2,)
                    or not np.isfinite(n).all()
                    or abs(np.linalg.norm(n) - 1) > 1e-5
                    or min(
                        observation["normal_uncertainty_px"], observation["tangent_uncertainty_px"]
                    )
                    <= 0
                ):
                    raise ValueError("Invalid directional source-boundary localization")

    def front_faces(self, rotation, translation):
        camera_local = (self.camera_t - translation) @ rotation
        direction = F.normalize(camera_local - self.face_center, dim=-1)
        return (self.face_normal * direction).sum(-1) > 1e-7

    def silhouette_mask(self, rotation, translation):
        return self.front_faces(rotation, translation)[self.edge_adjacent_faces].sum(-1) == 1

    def forward(self, human_vertices, object_R, object_t, frame_indices=None):
        if human_vertices.ndim != 3 or human_vertices.shape[1:] != (self.human_vertex_count, 3):
            raise ValueError("Supply current full original human mesh")
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
            raise ValueError("Invalid original source frame IDs")
        if object_R.shape != (len(frames), 3, 3) or object_t.shape != (len(frames), 3):
            raise ValueError("Object pose batch mismatch")
        values = []
        details = []
        if self.require_exterior_candidates:
            self.last_exterior_diagnostics = []
        for row, frame in enumerate(frames):
            exterior = None
            if self.require_exterior_candidates and any(
                a["frame"] == frame for a in self.observations
            ):
                try:
                    exterior = self._exterior_builder.build(
                        self.object_local,
                        self.object_faces,
                        object_R[row],
                        object_t[row],
                        self.camera_R,
                        self.camera_t,
                        self.focal,
                        self.image_center,
                        frame,
                    )
                finally:
                    if self._exterior_builder.last_audit is not None:
                        self.last_exterior_diagnostics.append(self._exterior_builder.last_audit)
            silhouette = self.silhouette_mask(object_R[row], object_t[row]).detach()
            for region, allowed in self.regions.items():
                observations = [
                    a for a in self.observations if a["frame"] == frame and a["region"] == region
                ]
                if not observations:
                    continue
                selected = torch.nonzero(silhouette & allowed.to(silhouette.device)).flatten()
                if not len(selected):
                    raise ValueError(
                        f"No live object silhouette for region {region}, frame {frame}; revise eligible source region explicitly"
                    )
                endpoints = (
                    self.object_local[self.edge_vertices[selected]] @ object_R[row].T
                    + object_t[row]
                )
                projected, depth = self.project(endpoints)
                # The projected front-face interior fixes the outward sign of
                # each curve normal. This prevents a distant horizontal edge
                # from substituting for a reviewed vertical boundary merely
                # because source tangential localization is less certain.
                adjacent = self.edge_adjacent_faces[selected]
                front = self.front_faces(object_R[row], object_t[row])
                front_id = torch.where(front[adjacent[:, 0]], adjacent[:, 0], adjacent[:, 1])
                center_uv, _ = self.project(
                    self.face_center[front_id] @ object_R[row].T + object_t[row]
                )
                edge_direction = F.normalize(projected[:, 1] - projected[:, 0], dim=-1)
                outward = torch.stack([-edge_direction[:, 1], edge_direction[:, 0]], -1)
                sign = ((center_uv - projected.mean(-2)) * outward).sum(-1) > 0
                outward = torch.where(sign[:, None], -outward, outward)
                observed = human_vertices.new_tensor([a["xy"] for a in observations])
                uncertainty = human_vertices.new_tensor(
                    [a["uncertainty_px"] for a in observations]
                )[:, None]
                if any("outward_image_normal_xy" in a for a in observations):
                    if not all("outward_image_normal_xy" in a for a in observations):
                        raise ValueError(
                            "Use consistent localization schema within one region/frame"
                        )
                    normal = human_vertices.new_tensor(
                        [a["outward_image_normal_xy"] for a in observations]
                    )
                    tangent = torch.stack([-normal[:, 1], normal[:, 0]], -1)
                    sigma = human_vertices.new_tensor(
                        [
                            [a["normal_uncertainty_px"], a["tangent_uncertainty_px"]]
                            for a in observations
                        ]
                    )
                    delta = projected[None] - observed[:, None, None]
                    whitened = (
                        torch.stack(
                            [
                                (delta * normal[:, None, None]).sum(-1),
                                (delta * tangent[:, None, None]).sum(-1),
                            ],
                            -1,
                        )
                        / sigma[:, None, None]
                    )
                    closest, beta = closest_projected_segment(
                        torch.zeros_like(observed[:, None]), whitened
                    )
                    normalized_error = closest.norm(dim=-1)
                    uv = (beta[..., None] * projected[None]).sum(-2)
                    normal_cosine = normal @ outward.T
                    orientation_ok = (normal_cosine >= self.min_boundary_normal_cosine).detach()
                else:
                    uv, beta = closest_projected_segment(observed[:, None], projected[None])
                    normalized_error = (uv - observed[:, None]).norm(dim=-1) / uncertainty
                    normal_cosine = torch.ones_like(normalized_error)
                    orientation_ok = torch.ones_like(normalized_error, dtype=torch.bool)
                alpha = beta / depth[None]
                alpha = alpha / alpha.sum(-1, keepdim=True)
                points = (alpha[..., None] * endpoints[None]).sum(-2)
                object_hit, human_hit = self._hits(
                    points, human_vertices[row], object_R[row], object_t[row], frame
                )
                object_tri = (
                    self.object_local[self.object_faces[object_hit.clamp_min(0)]] @ object_R[row].T
                    + object_t[row]
                )
                human_tri = human_vertices[row, self.human_faces[human_hit.clamp_min(0)]]
                do = ray_depth_excess(
                    points, self.camera_t, object_tri, object_hit >= 0, self.depth_tolerance_m
                )
                dh = ray_depth_excess(
                    points, self.camera_t, human_tri, human_hit >= 0, self.depth_tolerance_m
                )
                pixel = (uv - observed[:, None]).norm(dim=-1)
                huber = lambda x: torch.where(x < 1, 0.5 * x.square(), x - 0.5)
                cost = (
                    huber(normalized_error)
                    + self.pixel_guard_weight * F.relu(normalized_error - 1).square()
                    + self.occlusion_weight
                    * (huber(do / self.depth_scale_m) + huber(dh / self.depth_scale_m))
                )
                if self.normal_uncertainty_weight:
                    previous_cost = cost
                    cost = add_normal_uncertainty_cost(
                        cost, normal_cosine, observations, weight=self.normal_uncertainty_weight
                    )
                    normal_uncertainty_extra = cost - previous_cost
                prior = self.edge_length[selected].to(cost)
                prior = prior / prior.sum()
                logprior = prior.log()[None].expand_as(cost)
                visible = ((do == 0) & (dh == 0)).detach()
                eligible = orientation_ok & (
                    visible if self.require_visible_candidates else torch.ones_like(visible)
                )
                if exterior is not None:
                    exterior_distance = exterior.distances(uv)
                    exterior_ok = torch.as_tensor(exterior_distance <= 0.5, device=eligible.device)
                    eligible = eligible & exterior_ok
                if not bool(eligible.any(-1).all()):
                    unresolved = [
                        a.get("id", i)
                        for i, a in enumerate(observations)
                        if not bool(eligible[i].any())
                    ]
                    raise ValueError(
                        f"No visible current object boundary candidates with compatible direction at frame {frame}: {unresolved}. This is a local correspondence failure, not geometric infeasibility."
                    )
                logprior = logprior.masked_fill(~eligible, -torch.inf)
                logprior = logprior - torch.logsumexp(logprior, -1, keepdim=True)
                logmass = logprior - cost / self.temperature
                latent = -self.temperature * torch.logsumexp(logmass, -1)
                confidence = human_vertices.new_tensor([a["confidence"] for a in observations])
                values.extend((confidence * latent).unbind())
                with torch.no_grad():
                    posterior = logmass.softmax(-1)
                    chosen = posterior.argmax(-1)
                    for i, a in enumerate(observations):
                        k = chosen[i]
                        edge_id = selected[k]
                        bary = (alpha[i, k, :, None] * self.edge_barycentric[edge_id]).sum(0)
                        details.append(
                            dict(
                                id=a.get("id"),
                                frame=frame,
                                region=region,
                                selection_rule="Maximum posterior silhouette-edge diagnostic; not independent evidence",
                                original_object_face_id=int(self.edge_owner_face[edge_id]),
                                original_barycentric=bary.tolist(),
                                original_edge_vertex_ids=self.edge_vertices[edge_id].tolist(),
                                candidate_edges=len(selected),
                                visible_candidate_edges=int(visible[i].sum()),
                                require_visible_candidates=self.require_visible_candidates,
                                eligible_candidate_edges=int(eligible[i].sum()),
                                outward_normal_cosine=float(normal_cosine[i, k]),
                                posterior_mass=float(posterior[i, k]),
                                pixel_error_px=float(pixel[i, k]),
                                normalized_localization_error=float(normalized_error[i, k]),
                                object_occlusion_m=float(do[i, k]),
                                human_occlusion_m=float(dh[i, k]),
                                weighted_loss=float(confidence[i] * latent[i]),
                            )
                        )
                        if self.normal_uncertainty_weight:
                            details[-1].update(
                                normal_uncertainty_weight=self.normal_uncertainty_weight,
                                normal_direction_uncertainty_degrees=a.get(
                                    "normal_direction_uncertainty_degrees"
                                ),
                                normal_uncertainty_cost=float(normal_uncertainty_extra[i, k]),
                                normal_uncertainty_cost_is_approximate=True,
                            )
                        if exterior is not None:
                            details[-1].update(
                                require_exterior_candidates=True,
                                exterior_compatible_candidate_edges=int(exterior_ok[i].sum()),
                                exterior_selected_witness=exterior.selected_witness(uv[i, k]),
                                exterior_frame_audit=exterior.audit,
                            )
        self.last_details = details
        zero = human_vertices.sum() * 0 + object_R.sum() * 0 + object_t.sum() * 0
        return torch.stack(values).sum() / len(self.observations) + zero if values else zero
