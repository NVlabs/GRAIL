"""Foot-local continuity around reviewed support events on immutable terrain.

Stance no-slip never crosses an interval boundary. Transition windows penalize
motion corrections and excessive acceleration, not foot position or zero swing
velocity. Each interval retains its own fixed native material patch; no nearest
surface, new anchor, or event is inferred from the optimized motion.
"""

import math
from dataclasses import dataclass

import torch

from .scene_support import SceneSupportConstraint


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


def _envelope(positions, events, radius):
    result = torch.zeros_like(positions)
    for event in events:
        phase = (1 - (positions - event).abs() / radius).clamp(0, 1)
        result = torch.maximum(result, phase.square() * (3 - 2 * phase))
    return result


@dataclass(frozen=True)
class _Window:
    interval: object
    reference_center: torch.Tensor
    velocity_weights: torch.Tensor
    acceleration_weights: torch.Tensor


class SceneSupportTemporal:
    """Optional, explicitly weighted temporal terms for native scene support.

    ``no_slip`` is in metres and combines lag-1 with lag-4 displacement by
    default (not metres/second). The two window terms are dimensionless Huber
    penalties on excess speed/acceleration divided by their declared scales.
    Velocity stencils are located at frame midpoints; acceleration stencils at
    their central frame. Events are an interval's first supported frame and
    first unsupported frame, except sequence endpoints. Window envelopes are
    the smoothstep taper inside +/- ``window_frames`` source frames. No events
    means zero window penalties, even if stance no-slip remains applicable.
    """

    def __init__(
        self,
        reference_vertices,
        regions,
        manifest,
        *,
        fps,
        window_frames=4,
        velocity_scale_mps=0.35,
        acceleration_scale_mps2=3.0,
        correction_speed_tolerance_mps=0.15,
        acceleration_tolerance_mps2=3.0,
        normal_velocity_weight=0.25,
        long_horizon_lag=4,
        long_horizon_weight=0.25,
        tangent_blend_frames=3,
    ):
        self.fps = _number(fps, "fps", positive=True)
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
        self.normal_weight = _number(normal_velocity_weight, "normal_velocity_weight")
        self.long_weight = _number(long_horizon_weight, "long_horizon_weight")
        if type(window_frames) is not int or window_frames < 1:
            raise ValueError("window_frames must be a positive integer")
        if type(long_horizon_lag) is not int or long_horizon_lag < 2:
            raise ValueError("long_horizon_lag must be an integer >= 2")
        if type(tangent_blend_frames) is not int or tangent_blend_frames < 0:
            raise ValueError("tangent_blend_frames must be a nonnegative integer")
        self.long_lag = long_horizon_lag
        compiled = SceneSupportConstraint(
            reference_vertices, regions, manifest, tangent_blend_frames=tangent_blend_frames
        )
        self.shape, self.dtype, self.device = compiled.shape, compiled.dtype, compiled.device
        frame_count = self.shape[0]
        if frame_count < 2:
            raise ValueError("Temporal support requires at least two source frames")
        self.windows = []
        velocity_times = torch.arange(frame_count - 1, device=self.device, dtype=self.dtype) + 0.5
        acceleration_times = torch.arange(1, frame_count - 1, device=self.device, dtype=self.dtype)
        for interval in compiled.intervals:
            events = [interval.start] if interval.start > 0 else []
            if interval.end < frame_count:
                events.append(interval.end)
            self.windows.append(
                _Window(
                    interval,
                    reference_vertices[:, interval.ids].mean(dim=1).detach().clone(),
                    _envelope(velocity_times, events, window_frames),
                    _envelope(acceleration_times, events, window_frames),
                )
            )

    @staticmethod
    def _huber_nonnegative(value):
        return torch.where(value < 1, 0.5 * value.square(), value - 0.5)

    def terms(self, vertices):
        if (
            vertices.shape != self.shape
            or vertices.dtype != self.dtype
            or vertices.device != self.device
        ):
            raise ValueError("Temporal support native layout/device/dtype changed")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("Temporal support vertices must be finite")
        zero = vertices.sum() * 0.0
        totals = {key: zero for key in ("window_velocity", "window_acceleration")}
        denominators = {key: vertices.new_zeros(()) for key in totals}
        slips = {lag: [zero, vertices.new_zeros(())] for lag in (1, self.long_lag)}
        for window in self.windows:
            interval = window.interval
            center = vertices[:, interval.ids].mean(dim=1)
            stance = center[interval.start : interval.end]
            for lag, accumulator in slips.items():
                if len(stance) <= lag:
                    continue
                # Every point in the temporal stencil belongs to this same
                # authored interval; the boundary swing pair is never locked.
                weights = interval.tangent_weights.unfold(0, lag + 1, 1).amin(dim=-1)
                displacement = stance[lag:] - stance[:-lag]
                normal = (displacement @ interval.patch.normal)[:, None] * interval.patch.normal
                displacement = displacement - normal + self.normal_weight * normal
                accumulator[0] = accumulator[0] + (displacement.norm(dim=-1) * weights).sum()
                accumulator[1] = accumulator[1] + weights.sum()
            correction = center - window.reference_center
            correction_speed = (correction[1:] - correction[:-1]).norm(dim=-1) * self.fps
            acceleration = (center[2:] - 2 * center[1:-1] + center[:-2]).norm(dim=-1) * self.fps**2
            values = {
                "window_velocity": (
                    (correction_speed - self.speed_tolerance) / self.velocity_scale
                ).clamp_min(0),
                "window_acceleration": (
                    (acceleration - self.acceleration_tolerance) / self.acceleration_scale
                ).clamp_min(0),
            }
            for key, weights in (
                ("window_velocity", window.velocity_weights),
                ("window_acceleration", window.acceleration_weights),
            ):
                totals[key] = totals[key] + (self._huber_nonnegative(values[key]) * weights).sum()
                denominators[key] = denominators[key] + weights.sum()
        result = {key: totals[key] / denominators[key].clamp_min(1e-12) for key in totals}
        result["no_slip"] = slips[1][0] / slips[1][1].clamp_min(1e-12) + self.long_weight * slips[
            self.long_lag
        ][0] / slips[self.long_lag][1].clamp_min(1e-12)
        return result
