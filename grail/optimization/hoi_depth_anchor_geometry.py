"""Optional pure geometry for fixed initial-camera-ray depth hypotheses.

The isotropic anchor implementation is unchanged. These helpers only define an
explicit alternative residual: robust along-ray error plus quadratic lateral
error. They add no target, contact correspondence, camera or geometry changes.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from .hoi_depth_alignment import pseudo_huber_squared


def fixed_camera_ray_directions(initial_world, camera_centers):
    """Detached unit world rays from the actual calibrated camera center.

    ``camera_centers`` may have the same shape as the original material points
    or be a single XYZ center broadcast across them. Callers obtain centers from
    their existing calibrated cameras, never infer them from estimated depth.
    """
    if initial_world.ndim != 2 or initial_world.shape[1] != 3:
        raise ValueError("Initial original-material world points must be [N,3]")
    if camera_centers.shape not in (initial_world.shape, (3,), (1, 3)):
        raise ValueError("Camera centers must be one XYZ or one per original point")
    if initial_world.dtype != camera_centers.dtype or initial_world.device != camera_centers.device:
        raise ValueError("World points and calibrated camera centers must share dtype/device")
    if (
        not initial_world.is_floating_point()
        or not torch.isfinite(initial_world).all()
        or not torch.isfinite(camera_centers).all()
    ):
        raise ValueError("World points and camera centers must be finite floating tensors")
    delta = initial_world.detach() - camera_centers.detach()
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    if (length <= 1e-8).any():
        raise ValueError("Initial material point cannot coincide with camera center")
    return (delta / length).detach().clone()


def ray_separated_squared(
    current_world, target_world, fixed_unit_rays, *, huber_delta_m=0.03, lateral_weight=1.0
):
    """Per-point m² residual with a fixed, detached initial-ray hypothesis.

    Radial error is smooth pseudo-Huber squared; the world-space component
    perpendicular to the ray is quadratic. The target and ray receive no
    gradients. Lateral weighting does not change target depth or pixel ray.
    """
    if (
        current_world.ndim != 2
        or current_world.shape[1] != 3
        or target_world.shape != current_world.shape
        or fixed_unit_rays.shape != current_world.shape
    ):
        raise ValueError("Current, target and fixed rays must share shape [N,3]")
    if any(
        x.dtype != current_world.dtype or x.device != current_world.device
        for x in (target_world, fixed_unit_rays)
    ):
        raise ValueError("Residual inputs must share dtype and device")
    if not all(
        x.is_floating_point() and torch.isfinite(x).all()
        for x in (current_world, target_world, fixed_unit_rays)
    ):
        raise ValueError("Residual points and fixed rays must be finite floating tensors")
    if (
        isinstance(lateral_weight, bool)
        or not isinstance(lateral_weight, Real)
        or not math.isfinite(lateral_weight)
        or lateral_weight <= 0
    ):
        raise ValueError("Lateral weight must be finite and positive")
    rays = fixed_unit_rays.detach()
    if not torch.allclose(
        torch.linalg.vector_norm(rays, dim=-1),
        torch.ones(len(rays), device=rays.device, dtype=rays.dtype),
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("Fixed camera rays must be unit vectors")
    error = current_world - target_world.detach()
    radial = (error * rays).sum(-1)
    lateral = error - radial[:, None] * rays
    return pseudo_huber_squared(radial, huber_delta_m) + lateral_weight * lateral.square().sum(-1)


def initial_ray_pixel_consistency(world_points, sampled_native_uv, camera, max_error_px):
    """Reject inaccurate raster barycentrics without altering original triangles.

    The sampled image ray and the reconstructed original-material point must
    agree before that point is associated with estimated depth on that pixel.
    """
    if (
        isinstance(max_error_px, bool)
        or not isinstance(max_error_px, Real)
        or not math.isfinite(max_error_px)
        or max_error_px <= 0
    ):
        raise ValueError("Initial ray error threshold must be finite and positive")
    if (
        world_points.ndim != 2
        or world_points.shape[-1] != 3
        or sampled_native_uv.shape != (len(world_points), 2)
        or len(camera) != 1
    ):
        raise ValueError(
            "Expected [N,3] material points, [N,2] sampled pixels and one calibrated camera"
        )
    projected = camera.transform_points_screen(world_points[None])[0, :, :2]
    errors = torch.linalg.vector_norm(projected - sampled_native_uv, dim=-1)
    return torch.isfinite(errors) & (errors <= max_error_px), errors
