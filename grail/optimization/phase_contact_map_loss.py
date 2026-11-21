"""Partial contact between immutable, independently authored surface regions.

This is a latent regional objective, not a generated list of point pairs. Each
direction softly integrates the opposite region and covers a bounded fraction
of its own authored mass. The marginal cap prevents a single easy vertex from
explaining a broad contact map without requiring every candidate to touch.
Optional source-frame transitions mix complete regional objectives, never IDs
or geometry, and keep unknown/no-contact frames inactive.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contact_maps import PhaseContactMap


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _partial_mass(cost, prior, fraction, temperature):
    """Entropy-regularized capped marginal; use the envelope gradient.

    q minimizes <q,c> + temperature * KL(q||prior), with sum(q)=1 and
    q<=prior/fraction. The exact minimizer is clipped exponential tilting.
    Its detached value gives the envelope derivative, avoiding differentiation
    through the scalar root search. Zero-prior vertices keep zero mass.
    """
    with torch.no_grad():
        if fraction == 1:
            q = prior.expand_as(cost).clone()
        else:
            low = cost.min(-1, keepdim=True).values - 32 * temperature
            high = cost.max(-1, keepdim=True).values + temperature * math.log(1 / fraction)
            for _ in range(48):
                middle = (low + high) / 2
                mass = prior * torch.exp(((middle - cost) / temperature).clamp(max=math.log(1 / fraction)))
                lower = mass.sum(-1, keepdim=True) < 1
                low = torch.where(lower, middle, low)
                high = torch.where(lower, high, middle)
            q = prior * torch.exp((((low + high) / 2 - cost) / temperature).clamp(max=math.log(1 / fraction)))
    tiny = torch.finfo(cost.dtype).tiny
    kl = (q * (q.clamp_min(tiny).log() - prior.clamp_min(tiny).log())).sum(-1)
    return (q * cost).sum(-1) + temperature * kl, q


def _vertex_normals(vertices, faces):
    triangles = vertices[:, faces]
    normals = torch.cross(triangles[:, :, 1] - triangles[:, :, 0],
                          triangles[:, :, 2] - triangles[:, :, 0], dim=-1)
    total = torch.zeros_like(vertices)
    for corner in range(3):
        total = total.index_add(1, faces[:, corner], normals)
    length = torch.linalg.vector_norm(total, dim=-1)
    return F.normalize(total, dim=-1, eps=1e-12), length > 1e-12


def _quintic_ramp(length):
    """Reach unit activation over ``length`` source samples, never before onset."""
    u = np.arange(1, length + 1, dtype=np.float64) / length
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def _temporal_schedule(intervals, frames, activation_ramp_frames, phase_blend_frames):
    """Keep authored maps intact; only mix adjacent known-contact phase costs.

    The phase weights form a partition of unity inside each contact run. Onset
    activation multiplies that partition, so overlapping runtime windows cannot
    double contact force or leak through an unknown/released interval.
    """
    weights = np.zeros((len(intervals), frames), dtype=np.float64)
    activation = np.zeros(frames, dtype=np.float64)
    runs, blends = [], []
    i = 0
    while i < len(intervals):
        if intervals[i]["state"] != "contact":
            i += 1
            continue
        first = i
        while i < len(intervals) and intervals[i]["state"] == "contact":
            start, end = intervals[i]["frames"]
            weights[i, start:end] = 1
            i += 1
        start, end = intervals[first]["frames"][0], intervals[i - 1]["frames"][1]
        activation[start:end] = 1
        ramp_length = min(activation_ramp_frames, end - start)
        if ramp_length:
            activation[start:start + ramp_length] = _quintic_ramp(ramp_length)
        runs.append(dict(frames=[start, end], onset_window=[start, start + ramp_length],
                         requested_ramp_frames=activation_ramp_frames,
                         effective_ramp_frames=ramp_length))
    for i, interval in enumerate(intervals):
        if (not phase_blend_frames or i == 0 or interval["state"] != "contact"
                or intervals[i - 1]["state"] != "contact"):
            continue
        start, end = interval["frames"]
        length = min(phase_blend_frames, end - start)
        ramp = _quintic_ramp(length)
        weights[i, start:start + length] = ramp
        weights[i - 1, start:start + length] = 1 - ramp
        blends.append(dict(previous_phase=intervals[i - 1]["id"], phase=interval["id"],
                           frames=[start, start + length], requested_frames=phase_blend_frames,
                           effective_frames=length, incoming_weights=ramp.tolist()))
    weights *= activation[None]
    return weights, dict(contact_runs=runs, phase_blends=blends,
                         hand_activation=activation.tolist())


class _Region(nn.Module):
    def __init__(self, *, name, side, start, end, human_ids, object_ids, human_mass, object_mass,
                 authored_frames, phase_region_count, phase_weights):
        super().__init__()
        self.name, self.side, self.start, self.end = name, side, start, end
        self.authored_frames = authored_frames
        self.phase_region_count = phase_region_count
        self.register_buffer("phase_weights", torch.as_tensor(phase_weights, dtype=torch.float64))
        for key, value in (("human_ids", human_ids), ("object_ids", object_ids)):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.long))
        for key, value in (("human_mass", human_mass), ("object_mass", object_mass)):
            mass = torch.as_tensor(value, dtype=torch.float64)
            self.register_buffer(key, mass / mass.sum())


class PhaseContactMapLoss(nn.Module):
    @classmethod
    def from_file(cls, path, *, binding, object_vertices, object_faces, human_faces,
                  device=None, dtype=torch.float32, partial_mass_fraction=.25,
                  pair_temperature_m2=1.6e-5, coverage_temperature_m2=1.6e-5,
                  robust_delta_m=.01, gap_tolerance_m=.001,
                  normal_distance_scale_m=.005, min_opposing_cosine=.5,
                  activation_ramp_frames=0, phase_blend_frames=0):
        settings = dict(partial_mass_fraction=partial_mass_fraction,
                        pair_temperature_m2=pair_temperature_m2,
                        coverage_temperature_m2=coverage_temperature_m2,
                        robust_delta_m=robust_delta_m, gap_tolerance_m=gap_tolerance_m,
                        normal_distance_scale_m=normal_distance_scale_m,
                        min_opposing_cosine=min_opposing_cosine)
        for key, value in settings.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if not 0 < partial_mass_fraction <= 1:
            raise ValueError("partial_mass_fraction must lie in (0,1]")
        if min(pair_temperature_m2, coverage_temperature_m2, robust_delta_m) <= 0:
            raise ValueError("Temperatures and robust_delta_m must be positive")
        if min(gap_tolerance_m, normal_distance_scale_m) < 0 or not -1 <= min_opposing_cosine <= 1:
            raise ValueError("Invalid gap/normal settings")
        for key, value in (("activation_ramp_frames", activation_ramp_frames),
                           ("phase_blend_frames", phase_blend_frames)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer source-frame count")
            settings[key] = value
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Phase contact maps require float32 or float64 geometry")
        # The validated map owns a detached snapshot; preserve every original ID.
        maps = PhaseContactMap.from_file(path, binding=binding,
            object_vertices=_numpy(object_vertices), object_faces=_numpy(object_faces),
            human_faces=_numpy(human_faces))
        spec, local = maps.export_spec(), _numpy(object_vertices)
        regions, counts, mass_audit = [], np.zeros((2, binding["frame_count"])), []
        runtime_schedule = {}
        for side_index, side in enumerate(("left", "right")):
            intervals = spec["hands"][side + "_hand"]
            phase_weights, schedule = _temporal_schedule(intervals, binding["frame_count"],
                                                        activation_ramp_frames, phase_blend_frames)
            schedule["regions"] = []
            runtime_schedule[side] = schedule
            for interval_index, interval in enumerate(intervals):
                start, end = interval["frames"]
                counts[side_index, start:end] = len(interval["regions"])
                active = np.flatnonzero(phase_weights[interval_index] > 0)
                runtime_end = max(end, int(active[-1]) + 1) if len(active) else end
                for region in interval["regions"]:
                    ids = region["object_vertex_ids"]
                    # Exact native coincident vertices represent seams, not more
                    # evidence. Canonical geometry is hash-bound and unchanged.
                    _, inverse, frequency = np.unique(local[ids], axis=0, return_inverse=True, return_counts=True)
                    divisor = frequency[inverse]
                    regions.append(_Region(name=f"{side}/{interval['id']}/{region['id']}",
                        side=side, start=start, end=runtime_end,
                        human_ids=region["human_vertex_ids"], object_ids=ids,
                        human_mass=region["human_strength"],
                        object_mass=np.asarray(region["object_strength"]) / divisor,
                        authored_frames=[start, end], phase_region_count=len(interval["regions"]),
                        phase_weights=phase_weights[interval_index, start:runtime_end]))
                    schedule["regions"].append(dict(name=regions[-1].name,
                        authored_frames=[start, end], runtime_frames=[start, runtime_end],
                        phase_region_count=len(interval["regions"]),
                        effective_weights=(phase_weights[interval_index, start:runtime_end]
                                           / len(interval["regions"])).tolist()))
                    mass_audit.append(dict(region=regions[-1].name, object_native_count=len(ids),
                                           object_unique_positions=len(frequency),
                                           object_mass_divisors=divisor.tolist()))
        result = cls()
        result.regions = nn.ModuleList(regions)
        result.settings = settings
        result._spec = spec
        result._provenance = dict(schema="grail.phase_contact_map_loss.v1",
            contact_map=maps.provenance, optimizer_ready=True, contains_point_correspondences=False,
            objective="symmetric entropy-regularized capped partial regional coverage",
            settings=copy.deepcopy(settings),
            reduction="mean regions within authored phase; convex phase weights times contact-run activation; mean all source frames; sum hands",
            mass_semantics="authored strengths normalized within each region; exact coincident object IDs divide mass",
            mass_audit=mass_audit,
            temporal_policy="quintic onset within contact runs; convex adjacent-contact phase cost blend; no ID interpolation",
            runtime_schedule=runtime_schedule)
        result.register_buffer("object_local_vertices", torch.as_tensor(np.array(local, copy=True)))
        result.register_buffer("human_faces", torch.as_tensor(np.array(_numpy(human_faces), copy=True), dtype=torch.long))
        result.register_buffer("object_faces", torch.as_tensor(np.array(_numpy(object_faces), copy=True), dtype=torch.long))
        result.register_buffer("region_counts", torch.as_tensor(counts))
        return result.to(device=device, dtype=dtype)

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def export_spec(self):
        return copy.deepcopy(self._spec)

    def forward(self, human_vertices, object_R, object_t, return_details=False):
        binding = self._spec["binding"]
        frames = binding["frame_count"]
        if human_vertices.shape != (frames, binding["human_vertex_count"], 3):
            raise ValueError("Human vertices must preserve the bound full native topology/timeline")
        if object_R.shape != (frames, 3, 3) or object_t.shape != (frames, 3):
            raise ValueError("Object transform must preserve the bound full source timeline")
        if any(not torch.isfinite(value).all() for value in (human_vertices, object_R, object_t)):
            raise ValueError("Nonfinite live geometry/transform")
        if any(value.dtype != self.object_local_vertices.dtype or value.device != self.object_local_vertices.device
               for value in (human_vertices, object_R, object_t)):
            raise ValueError("Live geometry and contact helper must share dtype/device")
        identity = torch.eye(3, device=object_R.device, dtype=object_R.dtype)
        if (not torch.allclose(object_R.detach() @ object_R.detach().transpose(-1, -2),
                               identity.expand_as(object_R), atol=1e-4, rtol=1e-4)
                or torch.any(torch.linalg.det(object_R.detach()) <= 0)):
            raise ValueError("object_R must be a proper rotation, without extra scale")
        objects = self.object_local_vertices[None] @ object_R.transpose(-1, -2) + object_t[:, None]
        cfg = self.settings
        use_normals = cfg["normal_distance_scale_m"] > 0
        if use_normals:
            hn, hvalid = _vertex_normals(human_vertices, self.human_faces)
            on, ovalid = _vertex_normals(objects, self.object_faces)
        zero = (human_vertices.sum() + objects.sum()) * 0
        sides = dict(left=zero, right=zero)
        details = {}
        for region in self.regions:
            sl = slice(region.start, region.end)
            h, o = human_vertices[sl][:, region.human_ids], objects[sl][:, region.object_ids]
            a, b = region.human_mass[None], region.object_mass[None]
            gap = torch.linalg.vector_norm(h[:, :, None] - o[:, None], dim=-1)
            residual = F.relu(gap - cfg["gap_tolerance_m"])
            cost = cfg["robust_delta_m"] ** 2 * (torch.sqrt(1 + (residual / cfg["robust_delta_m"]) ** 2) - 1)
            opposition = None
            if use_normals:
                if (not hvalid[sl][:, region.human_ids][:, a[0] > 0].all()
                        or not ovalid[sl][:, region.object_ids][:, b[0] > 0].all()):
                    raise ValueError(f"Undefined native surface normals in {region.name}")
                opposition = -torch.einsum("fhi,foi->fho", hn[sl][:, region.human_ids], on[sl][:, region.object_ids])
                cost = cost + cfg["normal_distance_scale_m"] ** 2 * F.relu(cfg["min_opposing_cosine"] - opposition).square()
            tau = cfg["pair_temperature_m2"]
            loga = torch.where(a > 0, a.clamp_min(torch.finfo(a.dtype).tiny).log(), -torch.inf)
            logb = torch.where(b > 0, b.clamp_min(torch.finfo(b.dtype).tiny).log(), -torch.inf)
            hc = -tau * torch.logsumexp(logb[:, None] - cost / tau, dim=2)
            oc = -tau * torch.logsumexp(loga[:, :, None] - cost / tau, dim=1)
            hloss, hmass = _partial_mass(hc, a, cfg["partial_mass_fraction"], cfg["coverage_temperature_m2"])
            oloss, omass = _partial_mass(oc, b, cfg["partial_mass_fraction"], cfg["coverage_temperature_m2"])
            per_frame = (hloss + oloss) / 2
            if cfg["activation_ramp_frames"] or cfg["phase_blend_frames"]:
                # Normalize each authored phase first, then convex-mix phase
                # costs. Unequal region counts must not change the blend mass.
                contribution = per_frame / region.phase_region_count * region.phase_weights
            else:
                # Preserve the previous arithmetic and gradient path exactly.
                counts = self.region_counts[0 if region.side == "left" else 1, sl]
                contribution = per_frame / counts
            sides[region.side] = sides[region.side] + contribution.sum() / frames
            if return_details:
                with torch.no_grad():
                    hp = torch.softmax(logb[:, None] - cost / tau, dim=2) * hmass[:, :, None]
                    op = torch.softmax(loga[:, :, None] - cost / tau, dim=1) * omass[:, None]
                    posterior = (hp + op) / 2
                    record = dict(frames=[region.start, region.end],
                        authored_frames=region.authored_frames.copy(),
                        effective_weights=(region.phase_weights / region.phase_region_count).detach(),
                        regional_cost_m2=per_frame.detach(),
                        posterior_mean_gap_m=(posterior * gap).sum((1, 2)),
                        human_coverage_mass=hmass, object_coverage_mass=omass,
                        human_effective_count=1 / hmass.square().sum(1),
                        object_effective_count=1 / omass.square().sum(1),
                        human_max_mass=hmass.max(1).values, object_max_mass=omass.max(1).values,
                        max_pair_posterior=posterior.flatten(1).max(1).values,
                        posterior_gap_le_3mm=(posterior * (gap <= .003)).sum((1, 2)))
                    if opposition is not None:
                        record["posterior_opposing_cosine"] = (posterior * opposition).sum((1, 2))
                    details[region.name] = record
        total = sides["left"] + sides["right"]
        return dict(total=total, **sides, regions=details) if return_details else total
