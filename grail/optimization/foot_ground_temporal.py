"""Local touchdown/liftoff continuity from fixed native foot-contact labels.

These terms complement foot grounding and stance no-slip. They do not add
surface attraction, infer a floor, or impose zero swing velocity.
"""

import math

import torch

from grail.optimization.foot_ground import FOOT_PARTS


def _number(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _event_envelope(positions, events, radius):
    """Smoothstep windows with a maximum, rather than double counting overlaps."""
    result = torch.zeros_like(positions)
    for event in events:
        phase = (1 - (positions - event).abs() / radius).clamp(0, 1)
        result = torch.maximum(result, phase.square() * (3 - 2 * phase))
    return result


class FootGroundTemporal:
    """Freeze labels, material regions and reference centers once per fit.

    Probabilities are exactly [T,4] in L ankle/toe, R ankle/toe order. An event
    is the first frame whose strict ``probability > threshold`` state differs
    from the previous frame. Sequence endpoints are not invented events.

    Each event has a bounded +/- ``window_frames`` smoothstep window. Velocity
    stencils lie at frame midpoints and acceleration stencils at their central
    frame; contributing vertices may therefore extend one stencil sample beyond
    the window. Repeated/overlapping events never multiply a frame's weight.

    ``window_velocity`` penalizes excess speed of the *correction* to the fixed
    starting motion, preserving legitimate swing speed. ``window_acceleration``
    penalizes actual acceleration, including spikes already in that reference.
    Both use Huber penalties on excess magnitude divided by explicit scales,
    and source FPS supplies physical units. These are soft regularizers, not
    physical guarantees or substitutes for source/contact review.
    """

    def __init__(
        self,
        reference_vertices,
        regions,
        probabilities,
        *,
        fps,
        threshold=0.5,
        window_frames=4,
        velocity_scale_mps=0.35,
        acceleration_scale_mps2=3.0,
        correction_speed_tolerance_mps=0.15,
        acceleration_tolerance_mps2=3.0,
    ):
        self.fps = _number(fps, "fps", positive=True)
        self.threshold = _number(threshold, "threshold")
        if self.threshold > 1:
            raise ValueError("threshold must be in [0,1]")
        if type(window_frames) is not int or window_frames < 1:
            raise ValueError("window_frames must be a positive integer")
        self.velocity_scale = _number(velocity_scale_mps, "velocity_scale_mps", positive=True)
        self.acceleration_scale = _number(
            acceleration_scale_mps2, "acceleration_scale_mps2", positive=True
        )
        self.speed_tolerance = _number(
            correction_speed_tolerance_mps, "correction_speed_tolerance_mps"
        )
        self.acceleration_tolerance = _number(
            acceleration_tolerance_mps2, "acceleration_tolerance_mps2"
        )
        if (
            not isinstance(reference_vertices, torch.Tensor)
            or not reference_vertices.is_floating_point()
            or reference_vertices.ndim != 3
            or reference_vertices.shape[-1] != 3
            or not reference_vertices.shape[0]
            or not reference_vertices.shape[1]
            or not bool(torch.isfinite(reference_vertices).all())
        ):
            raise ValueError("Foot transitions require finite floating native vertices [T,V,3]")
        self.shape = reference_vertices.shape
        self.dtype, self.device = reference_vertices.dtype, reference_vertices.device
        if (
            not isinstance(probabilities, torch.Tensor)
            or probabilities.shape != (self.shape[0], 4)
            or probabilities.is_complex()
            or not bool(torch.isfinite(probabilities).all())
            or bool(((probabilities < 0) | (probabilities > 1)).any())
        ):
            raise ValueError("Foot transitions require finite fixed probabilities [T,4] in [0,1]")
        if len(regions) != 4:
            raise ValueError("Foot transitions require four native heel/toe regions")
        self.regions = []
        for label, ids in zip(FOOT_PARTS, regions):
            ids = torch.as_tensor(ids, device=self.device)
            if (
                ids.ndim != 1
                or not ids.numel()
                or ids.is_floating_point()
                or ids.is_complex()
                or ids.dtype == torch.bool
                or bool((ids < 0).any())
                or bool((ids >= self.shape[1]).any())
                or ids.unique().numel() != ids.numel()
            ):
                raise ValueError(f"Foot region {label} requires unique in-range native vertex IDs")
            self.regions.append(ids.detach().long().clone())
        all_ids = torch.cat(self.regions)
        if all_ids.unique().numel() != all_ids.numel():
            raise ValueError("Native heel/toe regions must be disjoint")
        self.regions = tuple(self.regions)
        self.reference_centers = self._centers(reference_vertices).detach().clone()
        self.active = (probabilities.detach().to(self.device) > self.threshold).clone()
        velocity_times = torch.arange(self.shape[0] - 1, device=self.device, dtype=self.dtype) + 0.5
        acceleration_times = torch.arange(
            1, max(self.shape[0] - 1, 1), device=self.device, dtype=self.dtype
        )
        self.events = tuple(
            tuple((torch.where(self.active[1:, part] != self.active[:-1, part])[0] + 1).tolist())
            for part in range(4)
        )
        self.velocity_weights = torch.stack(
            [_event_envelope(velocity_times, events, window_frames) for events in self.events],
            dim=1,
        )
        self.acceleration_weights = torch.stack(
            [_event_envelope(acceleration_times, events, window_frames) for events in self.events],
            dim=1,
        )

    def _centers(self, vertices):
        return torch.stack([vertices[:, ids].mean(dim=1) for ids in self.regions], dim=1)

    @staticmethod
    def _penalty(magnitude, tolerance, scale, weights):
        excess = ((magnitude - tolerance) / scale).clamp_min(0)
        robust = torch.where(excess < 1, 0.5 * excess.square(), excess - 0.5)
        return (robust * weights).sum() / weights.sum().clamp_min(1e-12)

    def terms(self, vertices):
        if (
            not isinstance(vertices, torch.Tensor)
            or vertices.shape != self.shape
            or vertices.dtype != self.dtype
            or vertices.device != self.device
        ):
            raise ValueError("Foot-transition native layout/device/dtype changed within a fit")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("Foot-transition vertices must be finite")
        centers = self._centers(vertices)
        correction = centers - self.reference_centers
        speed = correction.diff(dim=0).norm(dim=-1) * self.fps
        acceleration = centers.diff(n=2, dim=0).norm(dim=-1) * self.fps**2
        return {
            "window_velocity": self._penalty(
                speed, self.speed_tolerance, self.velocity_scale, self.velocity_weights
            ),
            "window_acceleration": self._penalty(
                acceleration,
                self.acceleration_tolerance,
                self.acceleration_scale,
                self.acceleration_weights,
            ),
        }
