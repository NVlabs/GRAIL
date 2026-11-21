"""Source-window temporal controls for independently authored contact maps.

These terms preserve native material motion, not a framewise contact posterior.
Stable holds must be named explicitly; no slip constraint is inferred from an
interval's name. Local world-motion windows are also explicit and may cover an
approach/contact join without adding contact attraction before first touch.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class _Field(nn.Module):
    def __init__(self, region):
        super().__init__()
        self.name, self.side = region.name, region.side
        self.register_buffer("ids", region.human_ids.detach().clone())
        self.register_buffer("mass", region.human_mass.detach().clone())


def _finite_number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _window_envelope(length, ramp_frames, *, dtype, device=None):
    """Frame, edge and triple weights; ramp 0 preserves a hard full window."""
    if ramp_frames:
        distance = torch.minimum(torch.arange(length, dtype=dtype, device=device),
                                 torch.arange(length - 1, -1, -1, dtype=dtype, device=device))
        u = (distance / ramp_frames).clamp(max=1)
        frame = u.pow(3) * (10 - 15 * u + 6 * u.square())
    else:
        frame = torch.ones(length, dtype=dtype, device=device)
    edge = torch.minimum(frame[:-1], frame[1:])
    triple = torch.minimum(edge[:-1], edge[1:])
    return frame, edge, triple


class PhaseContactTemporalLoss(nn.Module):
    """Return three dimensionless terms, weighted separately by the caller.

    ``hold_interval_ids`` maps ``left``/``right`` to reviewed stable-contact
    interval IDs. Only differences wholly within each named interval contribute.
    ``windows`` consists of dicts requiring ``id``, ``frames`` (half-open),
    ``regions`` (full ``side/interval/region`` names), and ``object`` (bool).
    Optional ``ramp_frames=0`` adds a quintic fade from each endpoint over that
    many source-frame intervals; stencil weights are the minimum constituent
    frame weights. ``velocity_weight=1`` and ``acceleration_weight=1`` scale each
    window's terms. Reduction remains the arithmetic mean over declared windows,
    with no normalization by the sum of weights or the effective fade mass.
    Region material IDs/strengths stay fixed throughout each window, including
    any source-supported join it covers. This does not activate contact there.

    Each native point's speed/acceleration contributes a squared hinge above its
    configured tolerance, divided by the corresponding explicit unit scale.
    Thus rotations and local deformation cannot cancel through a centroid.
    Object motion uses lumped original-triangle-area weights on native vertices.
    The terms are regularizers; they do not impose hard speed limits or certify
    contact, collision freedom, or physical validity.
    """

    def __init__(self, contact_helper, *, fps, hold_interval_ids, windows,
                 hold_speed_tolerance_mps=.02, hold_speed_scale_mps=.05,
                 window_speed_tolerance_mps=.3, window_speed_scale_mps=.3,
                 window_acceleration_tolerance_mps2=3.,
                 window_acceleration_scale_mps2=3., hold_ramp_frames=0):
        super().__init__()
        settings = dict(fps=fps, hold_speed_tolerance_mps=hold_speed_tolerance_mps,
            hold_speed_scale_mps=hold_speed_scale_mps,
            window_speed_tolerance_mps=window_speed_tolerance_mps,
            window_speed_scale_mps=window_speed_scale_mps,
            window_acceleration_tolerance_mps2=window_acceleration_tolerance_mps2,
            window_acceleration_scale_mps2=window_acceleration_scale_mps2)
        for key, value in settings.items():
            _finite_number(value, key, positive=(key == "fps" or "scale" in key))
        if type(hold_ramp_frames) is not int or hold_ramp_frames < 0:
            raise ValueError("hold_ramp_frames must be a nonnegative integer")
        spec = contact_helper.export_spec()
        self._binding = copy.deepcopy(spec["binding"])
        if not math.isclose(fps, self._binding["fps"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("fps must match the bound source timeline")
        if not isinstance(hold_interval_ids, dict) or set(hold_interval_ids) != {"left", "right"}:
            raise ValueError("hold_interval_ids must explicitly contain left and right")
        self.fields = nn.ModuleList([_Field(region) for region in contact_helper.regions])
        lookup = {field.name: i for i, field in enumerate(self.fields)}
        self._holds = []
        for side in ("left", "right"):
            ids = hold_interval_ids[side]
            if (not isinstance(ids, list) or any(not isinstance(x, str) for x in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("Hold interval IDs must be unique lists of strings")
            intervals = {item["id"]: item for item in spec["hands"][side + "_hand"]}
            for interval_id in ids:
                if interval_id not in intervals or intervals[interval_id]["state"] != "contact":
                    raise ValueError(f"Hold interval must explicitly name contact: {side}/{interval_id}")
                interval = intervals[interval_id]
                start, end = interval["frames"]
                if end - start < 2:
                    raise ValueError("A stable hold needs at least two source frames")
                names = [f"{side}/{interval_id}/{r['id']}" for r in interval["regions"]]
                self._holds.append(dict(side=side, interval=interval_id, frames=[start, end],
                                        fields=[lookup[name] for name in names]))
        if not isinstance(windows, list):
            raise ValueError("windows must be an explicit list")
        self._windows = []
        seen = set()
        for window in windows:
            required = {"id", "frames", "regions", "object"}
            optional = {"ramp_frames", "velocity_weight", "acceleration_weight"}
            if (not isinstance(window, dict) or not required <= set(window)
                    or set(window) - required - optional):
                raise ValueError("Each window requires id, frames, regions, object; optional ramp_frames and derivative weights")
            name, frames, names = window["id"], window["frames"], window["regions"]
            if not isinstance(name, str) or not name.strip() or name in seen:
                raise ValueError("Window IDs must be nonempty unique strings")
            seen.add(name)
            if (not isinstance(frames, list) or len(frames) != 2
                    or any(type(x) is not int for x in frames)
                    or not 0 <= frames[0] < frames[1] <= self._binding["frame_count"]
                    or frames[1] - frames[0] < 3):
                raise ValueError("Window frames must be a valid half-open interval of at least three source frames")
            if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
                    or len(names) != len(set(names)) or any(n not in lookup for n in names)):
                raise ValueError("Window regions must explicitly reference unique bound region names")
            if type(window["object"]) is not bool or (not names and not window["object"]):
                raise ValueError("Window must select a human region or the object")
            ramp = window.get("ramp_frames", 0)
            if type(ramp) is not int or ramp < 0:
                raise ValueError("Window ramp_frames must be a nonnegative integer")
            vw, aw = window.get("velocity_weight", 1.), window.get("acceleration_weight", 1.)
            _finite_number(vw, "window velocity_weight")
            _finite_number(aw, "window acceleration_weight")
            _, edge, triple = _window_envelope(frames[1] - frames[0], ramp, dtype=torch.float64)
            if not (vw * edge.sum() + aw * triple.sum() > 0):
                raise ValueError("Window must retain at least one positive effective derivative weight")
            self._windows.append(dict(id=name, frames=frames.copy(),
                fields=[lookup[n] for n in names], object=window["object"],
                ramp_frames=ramp, velocity_weight=vw, acceleration_weight=aw))
        local = contact_helper.object_local_vertices.detach().clone()
        faces = contact_helper.object_faces
        triangles = local[faces]
        areas = torch.linalg.vector_norm(torch.cross(triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1) / 2
        mass = local.new_zeros(len(local))
        for corner in range(3):
            mass.index_add_(0, faces[:, corner], areas / 3)
        if not torch.isfinite(mass).all() or mass.sum() <= 0:
            raise ValueError("Original object surface needs finite nonzero triangle area")
        self.register_buffer("object_local_vertices", local)
        self.register_buffer("object_mass", mass / mass.sum())
        self.settings = dict(settings, hold_ramp_frames=hold_ramp_frames)
        window_weights = []
        for window in self._windows:
            start, end = window["frames"]
            frame, edge, triple = _window_envelope(end - start, window["ramp_frames"], dtype=torch.float64)
            window_weights.append(dict(id=window["id"], frames=[start, end],
                ramp_frames=window["ramp_frames"], velocity_weight=window["velocity_weight"],
                acceleration_weight=window["acceleration_weight"], frame_envelope=frame.tolist(),
                effective_velocity_edge_weights=(edge * window["velocity_weight"] / (len(edge) * len(self._windows))).tolist(),
                effective_acceleration_triple_weights=(triple * window["acceleration_weight"] / (len(triple) * len(self._windows))).tolist()))
        self._provenance = dict(schema="grail.phase_contact_temporal.v1",
            contact_map=contact_helper.provenance["contact_map"],
            settings=copy.deepcopy(self.settings),
            hold_interval_ids=copy.deepcopy(hold_interval_ids), windows=copy.deepcopy(windows),
            window_weights=window_weights,
            contains_point_correspondences=False,
            material_weights="fixed authored human strengths; original triangle-area lumped object vertices",
            frame_semantics="half-open source intervals; hold differences never cross interval boundaries",
            reduction="mean source differences and regions within interval; mean intervals per hand then sum hands; "
                      "window terms mean regions per hand, equal mean participating hands/object, then coefficient-weighted mean windows; "
                      "no normalization by coefficient sum or effective fade mass",
            units="dimensionless squared hinge; caller applies separate term weights",
            window_ramp="quintic frame envelope, zero at both endpoints for positive ramp_frames; "
                        "minimum constituent-frame weights for velocity edges and acceleration triples",
            hold_ramp="optional cubic smoothstep over first hold_ramp_frames velocity edges; no temporal extrapolation")

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def _penalty(self, derivative, mass, tolerance, scale):
        speed = torch.linalg.vector_norm(derivative, dim=-1)
        per_edge = (F.relu(speed - tolerance).square() * mass).sum(-1) / scale ** 2
        return per_edge, speed

    def forward(self, human_vertices, object_R, object_t, return_details=False):
        count = self._binding["frame_count"]
        if human_vertices.shape != (count, self._binding["human_vertex_count"], 3):
            raise ValueError("Human vertices must preserve bound full native topology/timeline")
        if object_R.shape != (count, 3, 3) or object_t.shape != (count, 3):
            raise ValueError("Object transform must preserve full source timeline")
        for value in (human_vertices, object_R, object_t):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite live geometry/transform")
            if value.dtype != self.object_local_vertices.dtype or value.device != self.object_local_vertices.device:
                raise ValueError("Live geometry and temporal helper must share dtype/device")
        identity = torch.eye(3, dtype=object_R.dtype, device=object_R.device)
        if (not torch.allclose(object_R.detach() @ object_R.detach().transpose(-1, -2),
                              identity.expand_as(object_R), atol=1e-4, rtol=1e-4)
                or torch.any(torch.linalg.det(object_R.detach()) <= 0)):
            raise ValueError("object_R must be a proper rotation")
        cfg, fps = self.settings, self.settings["fps"]
        zero = (human_vertices.sum() + object_R.sum() + object_t.sum()) * 0
        holds = {"left": [], "right": []}
        details = {"holds": {}, "windows": {}}
        for interval in self._holds:
            start, end = interval["frames"]
            values = []
            for index in interval["fields"]:
                field = self.fields[index]
                h = human_vertices[start:end][:, field.ids]
                local = (h - object_t[start:end, None]) @ object_R[start:end]
                penalty, speed = self._penalty(local.diff(dim=0) * fps, field.mass,
                    cfg["hold_speed_tolerance_mps"], cfg["hold_speed_scale_mps"])
                if cfg["hold_ramp_frames"]:
                    u = (torch.arange(1, len(penalty) + 1, dtype=h.dtype, device=h.device)
                         / cfg["hold_ramp_frames"]).clamp(max=1)
                    penalty = penalty * (u.square() * (3 - 2 * u))
                values.append(penalty.mean())
                if return_details:
                    details["holds"][field.name] = dict(frames=[start, end],
                        weighted_speed_mps=(speed * field.mass).sum(-1).detach(),
                        penalty=penalty.detach())
            holds[interval["side"]].append(torch.stack(values).mean())
        hold_loss = sum((torch.stack(values).mean() if values else zero) for values in holds.values())
        velocity, acceleration = [], []
        for window in self._windows:
            start, end = window["frames"]
            envelope, edge_weight, triple_weight = _window_envelope(end - start,
                window["ramp_frames"], dtype=human_vertices.dtype, device=human_vertices.device)
            subjects = {"left": [], "right": [], "object": []}
            records = {}
            for index in window["fields"]:
                field = self.fields[index]
                subjects[field.side].append((field.name, human_vertices[start:end][:, field.ids], field.mass))
            if window["object"]:
                obj = self.object_local_vertices[None] @ object_R[start:end].transpose(-1, -2) + object_t[start:end, None]
                subjects["object"].append(("object", obj, self.object_mass))
            subject_v, subject_a = [], []
            for fields in subjects.values():
                vterms, aterms = [], []
                for name, points, mass in fields:
                    v = points.diff(dim=0) * fps
                    a = v.diff(dim=0) * fps
                    vp, speed = self._penalty(v, mass, cfg["window_speed_tolerance_mps"], cfg["window_speed_scale_mps"])
                    ap, accel = self._penalty(a, mass, cfg["window_acceleration_tolerance_mps2"], cfg["window_acceleration_scale_mps2"])
                    vp = vp * edge_weight * window["velocity_weight"]
                    ap = ap * triple_weight * window["acceleration_weight"]
                    vterms.append(vp.mean())
                    aterms.append(ap.mean())
                    if return_details:
                        records[name] = dict(weighted_speed_mps=(speed * mass).sum(-1).detach(),
                            weighted_acceleration_mps2=(accel * mass).sum(-1).detach(),
                            velocity_penalty=vp.detach(), acceleration_penalty=ap.detach())
                if fields:
                    subject_v.append(torch.stack(vterms).mean())
                    subject_a.append(torch.stack(aterms).mean())
            velocity.append(torch.stack(subject_v).mean())
            acceleration.append(torch.stack(subject_a).mean())
            if return_details:
                details["windows"][window["id"]] = dict(frames=[start, end], subjects=records,
                    frame_envelope=envelope.detach(),
                    velocity_edge_weights=(edge_weight * window["velocity_weight"]).detach(),
                    acceleration_triple_weights=(triple_weight * window["acceleration_weight"]).detach())
        result = dict(hold_relative_velocity=hold_loss,
                      window_velocity=torch.stack(velocity).mean() if velocity else zero,
                      window_acceleration=torch.stack(acceleration).mean() if acceleration else zero)
        if return_details:
            result["details"] = details
        return result
