"""Optional current-human/current-human latent support diagnostic, schema v001.

No LC registration or file loading. Original faces/area/priors are explicit inputs.
Gap and orientation are separate unit losses: they may choose different pairs.
Neither is a collision, object-blocker, visibility, force or contact certificate.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "current_human_region_support_v001"


@dataclass(frozen=True)
class SupportOptions:
    gap_tolerance_m: float = 0.001
    robust_delta_m: float = 0.01
    gap_temperature_m2: float = 0.000016
    min_opposing_cosine: float = 0.5
    orientation_temperature: float = 0.05
    max_edge_m: float = 0.002
    orientation_distance_scale_m: float | None = None


def latent_pair_cost(cost, left_mass, right_mass, temperature):
    """Area-weighted normalized alternatives; subdividing mass cannot add force."""
    if cost.shape[-2:] != (left_mass.numel(), right_mass.numel()):
        raise ValueError("Pair cost and mass shape mismatch")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive finite latent temperature required")
    for mass in (left_mass, right_mass):
        if (
            mass.ndim != 1
            or not bool(torch.isfinite(mass).all())
            or bool((mass < 0).any())
            or not bool(mass.sum() > 0)
        ):
            raise ValueError("Positive represented prior mass required")
    p = left_mass / left_mass.sum()
    q = right_mass / right_mass.sum()
    prior = p[:, None] * q[None]
    logp = torch.where(prior > 0, prior.clamp_min(torch.finfo(cost.dtype).tiny).log(), -torch.inf)
    return -temperature * torch.logsumexp((logp - cost / temperature).flatten(-2), -1)


class _Patch(nn.Module):
    def __init__(self, rest, faces, face_ids, vertex_prior, max_edge_m):
        super().__init__()
        import numpy as np

        from grail.optimization.visible_skin import canonical_surface_quadrature

        ids = np.asarray(face_ids)
        prior = torch.as_tensor(vertex_prior, dtype=torch.float64)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not len(ids)
            or len(np.unique(ids)) != len(ids)
            or ids.min() < 0
            or ids.max() >= len(faces)
        ):
            raise ValueError("Unique nonempty original face IDs required")
        if (
            prior.shape != (len(rest),)
            or not bool(torch.isfinite(prior).all())
            or bool((prior < 0).any())
        ):
            raise ValueError("Finite nonnegative original-vertex prior required")
        fi, bary, area, info = canonical_surface_quadrature(rest, faces, ids, max_edge_m)
        tri = torch.as_tensor(faces[fi], dtype=torch.long)
        bary = torch.as_tensor(bary, dtype=torch.float64)
        area = torch.as_tensor(area, dtype=torch.float64)
        mass = area * (prior[tri] * bary).sum(-1)
        if not bool(mass.sum() > 0):
            raise ValueError("Region has no supported represented area")
        self.register_buffer("face_ids", torch.as_tensor(fi, dtype=torch.long))
        self.register_buffer("triangles", tri)
        self.register_buffer("barycentric", bary)
        self.register_buffer("area_m2", area)
        self.register_buffer("mass_m2", mass)
        self.info = dict(
            info,
            original_region_face_ids=ids.tolist(),
            represented_area_m2=float(area.sum()),
            prior_mass_m2=float(mass.sum()),
        )

    def current(self, vertices):
        tri = vertices[:, self.triangles]
        pos = (tri * self.barycentric[None, :, :, None]).sum(-2)
        cross = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
        lengths = torch.linalg.vector_norm(cross, dim=-1)
        if not bool(torch.isfinite(lengths).all()) or bool((lengths <= 1e-14).any()):
            raise ValueError("Current supported triangle is degenerate; no invented normal")
        return pos, cross / lengths[..., None]


class HumanRegionSupportLoss(nn.Module):
    """One shared event E, normalized alternative interfaces, original surfaces.

    interface_strengths[F,I] are graded relationship strengths, not independent
    left/right probabilities. E=max(strengths); alpha=strengths/sum(strengths).
    Loss=mean_frames(E * sum_i(alpha_i * latent_i)). Unknown E=0 contributes0
    with the fixed frame denominator. Both side priors contain spatial anatomy
    only; confidence cannot be multiplied again inside their normalized mass.

    Optional orientation_distance_scale_m explicitly converts the normal penalty
    into m^2 for ONE coupled pair-cost posterior. It has no selected default;
    independent diagnostics alone must not be used to assert joint agreement.

    Image localization uncertainty stays in its source-image term. This helper
    has no camera, source pixel, image uncertainty or hidden pad-normal input.
    """

    def __init__(
        self,
        *,
        human_rest_vertices,
        human_faces,
        interfaces,
        interface_strengths,
        enabled=False,
        options=None,
    ):
        super().__init__()
        if not isinstance(enabled, bool):
            raise ValueError("Explicit enabled boolean required")
        self.enabled = enabled
        self.options = options or SupportOptions()
        self.patches = nn.ModuleDict()
        self.names = []
        self.provenance = {"schema": SCHEMA, "enabled": enabled, "saved_world_targets": False}
        if not enabled:
            return
        import numpy as np

        rest, faces = np.asarray(human_rest_vertices), np.asarray(human_faces)
        if (
            rest.ndim != 2
            or rest.shape[1] != 3
            or not np.isfinite(rest).all()
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or not len(faces)
            or faces.min() < 0
            or faces.max() >= len(rest)
        ):
            raise ValueError("Finite original topology required")
        o = self.options
        if (
            not all(math.isfinite(x) for x in vars(o).values() if x is not None)
            or min(o.robust_delta_m, o.gap_temperature_m2, o.orientation_temperature, o.max_edge_m)
            <= 0
            or o.max_edge_m > 0.002
            or o.gap_tolerance_m < 0
            or not -1 <= o.min_opposing_cosine <= 1
            or (o.orientation_distance_scale_m is not None and o.orientation_distance_scale_m < 0)
        ):
            raise ValueError("Invalid support options")
        strengths = torch.as_tensor(interface_strengths, dtype=torch.float64)
        if (
            strengths.ndim != 2
            or strengths.shape[1] != len(interfaces)
            or not len(interfaces)
            or not len(strengths)
            or not bool(torch.isfinite(strengths).all())
            or bool(((strengths < 0) | (strengths > 1)).any())
        ):
            raise ValueError("Expected frame/interface relationship strengths in[0,1]")
        self.register_buffer("strengths", strengths)
        self.vertex_count = len(rest)
        for item in interfaces:
            name = item["name"]
            if not isinstance(name, str) or not name or "." in name or name in self.names:
                raise ValueError("Unique simple interface names required")
            self.names.append(name)
            left_ids, right_ids = [np.asarray(item[s + "_face_ids"]) for s in ("left", "right")]
            for ids in (left_ids, right_ids):
                if (
                    ids.ndim != 1
                    or ids.dtype.kind not in "iu"
                    or not len(ids)
                    or len(np.unique(ids)) != len(ids)
                    or ids.min() < 0
                    or ids.max() >= len(faces)
                ):
                    raise ValueError("Validate original face IDs before topology indexing")
            if np.intersect1d(faces[left_ids].ravel(), faces[right_ids].ravel()).size:
                raise ValueError("Support sides must have disjoint original vertices")
            for side in ("left", "right"):
                self.patches[name + "_" + side] = _Patch(
                    rest,
                    faces,
                    item[side + "_face_ids"],
                    item[side + "_vertex_prior"],
                    o.max_edge_m,
                )
        self.provenance.update(
            interfaces={
                n: {s: self.patches[n + "_" + s].info for s in ("left", "right")}
                for n in self.names
            },
            strength_rule="E=max strength; alpha=strength/sum; apply E once; fixed input-frame mean",
            gap_units="m^2",
            orientation_units="dimensionless",
            normal_source="Current original triangle orientation; not an observed pad normal",
            localization="No image localization is inferred or implemented",
            limitations="Different gap/orientation latent winners; alternatives may share target patches; no exclusivity, collision, blocker or visibility guarantee",
        )

    def forward(self, human_vertices, *, frame_indices):
        zero = human_vertices.sum() * 0
        if not self.enabled:
            return {"gap_m2": zero, "orientation": zero, "interfaces": {}, "enabled": False}
        if (
            human_vertices.ndim != 3
            or human_vertices.shape[1:] != (self.vertex_count, 3)
            or not len(human_vertices)
            or not bool(torch.isfinite(human_vertices).all())
        ):
            raise ValueError("Current full original human mesh required")
        supplied = torch.as_tensor(frame_indices, device=self.strengths.device)
        if (
            supplied.dtype == torch.bool
            or supplied.is_complex()
            or not bool(torch.isfinite(supplied).all())
            or (supplied.is_floating_point() and not bool((supplied == supplied.round()).all()))
        ):
            raise ValueError("Original frame IDs must be integral before conversion")
        frames = supplied.to(torch.long)
        if (
            frames.shape != (len(human_vertices),)
            or bool(((frames < 0) | (frames >= len(self.strengths))).any())
            or len(torch.unique(frames)) != len(frames)
        ):
            raise ValueError("Unique original frame IDs required")
        strengths = self.strengths[frames]
        E = strengths.max(-1).values
        alpha = strengths / strengths.sum(-1, keepdim=True).clamp_min(
            torch.finfo(strengths.dtype).tiny
        )
        gaps, orientations, joint_values, details = [], [], [], {}
        for i, name in enumerate(self.names):
            active = strengths[:, i] > 0
            g, n = zero.expand(len(frames)).clone(), zero.expand(len(frames)).clone()
            joint = zero.expand(len(frames)).clone()
            joint_summary = None
            if bool(active.any()):
                lp, rp = [self.patches[name + "_" + s] for s in ("left", "right")]
                L, Ln = lp.current(human_vertices[active])
                R, Rn = rp.current(human_vertices[active])
                distance = torch.linalg.vector_norm(L[:, :, None] - R[:, None], dim=-1)
                excess = F.relu(distance - self.options.gap_tolerance_m)
                delta = self.options.robust_delta_m
                cost = delta**2 * (torch.sqrt(1 + (excess / delta) ** 2) - 1)
                opposition = -torch.einsum("fli,fri->flr", Ln, Rn)
                nc = F.relu(self.options.min_opposing_cosine - opposition).square()
                g[active] = latent_pair_cost(
                    cost, lp.mass_m2, rp.mass_m2, self.options.gap_temperature_m2
                )
                n[active] = latent_pair_cost(
                    nc, lp.mass_m2, rp.mass_m2, self.options.orientation_temperature
                )
                if self.options.orientation_distance_scale_m is not None:
                    joint_cost = cost + self.options.orientation_distance_scale_m**2 * nc
                    joint[active] = latent_pair_cost(
                        joint_cost, lp.mass_m2, rp.mass_m2, self.options.gap_temperature_m2
                    )
                    with torch.no_grad():
                        p = lp.mass_m2 / lp.mass_m2.sum()
                        q = rp.mass_m2 / rp.mass_m2.sum()
                        mass = p[:, None] * q[None]
                        logprior = torch.where(
                            mass > 0,
                            mass.clamp_min(torch.finfo(joint_cost.dtype).tiny).log(),
                            -torch.inf,
                        )
                        posterior = torch.softmax(
                            (logprior - joint_cost / self.options.gap_temperature_m2).flatten(1), -1
                        ).reshape_as(joint_cost)
                        joint_summary = {
                            "posterior_mean_gap_m": (posterior * distance).sum((1, 2)),
                            "posterior_mean_opposing_cosine": (posterior * opposition).sum((1, 2)),
                            "max_pair_posterior": posterior.flatten(1).max(-1).values,
                        }
                details[name] = {
                    "joint_m2": (
                        joint.detach()
                        if self.options.orientation_distance_scale_m is not None
                        else None
                    ),
                    "joint_posterior": joint_summary,
                    "minimum_gap_m": distance.detach().flatten(1).min(-1).values,
                    "gap_m2": g.detach(),
                    "orientation": n.detach(),
                    "active": active.detach(),
                    "event_confidence": E.detach(),
                    "normalized_alternative_weight": alpha[:, i].detach(),
                }
            gaps.append(g)
            orientations.append(n)
            joint_values.append(joint)
        gap = (E * (alpha * torch.stack(gaps, -1)).sum(-1)).mean()
        orient = (E * (alpha * torch.stack(orientations, -1)).sum(-1)).mean()
        coupled = (
            None
            if self.options.orientation_distance_scale_m is None
            else (E * (alpha * torch.stack(joint_values, -1)).sum(-1)).mean()
        )
        return {
            "gap_m2": gap,
            "orientation": orient,
            "joint_m2": coupled,
            "interfaces": details,
            "enabled": True,
        }
