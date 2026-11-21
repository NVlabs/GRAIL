"""Physical-time priors on HOI trajectory corrections, not contact discovery.

These optional experimental terms act over the entire source timeline. They
penalize new motion introduced by fitting; inherited initializer spikes still
need independent native-motion review. No hand/object attraction is added in
inactive or uncertain frames.
"""

from __future__ import annotations

import math
from numbers import Real


def validate_motion_options(options):
    defaults = {
        "velocity_scale_m_s": 0.25,
        "acceleration_scale_m_s2": 3.0,
        "jerk_scale_m_s3": 40.0,
        "peak_weight": 0.1,
    }
    if set(options) - defaults.keys():
        raise ValueError("Unknown reviewed motion option")
    values = {**defaults, **options}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite number")
        if not math.isfinite(value) or (value < 0 if key == "peak_weight" else value <= 0):
            raise ValueError(f"Invalid {key}")
    return values


def correction_motion_loss(
    current,
    reference,
    *,
    fps,
    velocity_scale_m_s=0.25,
    acceleration_scale_m_s2=3.0,
    jerk_scale_m_s3=40.0,
    peak_weight=0.1,
):
    """Mean + peak squared v/a/j of current-reference, scaled by physical units.

    Both inputs have shape (frames, tracked_points, 3) in metres. Reference is
    detached. Constant spatial offsets cost zero; source projection/depth and
    pose terms remain necessary. A max term prevents one bad transition from
    disappearing in an average over frames or points. This is a soft objective,
    not a hard bound or a guarantee of smooth output.
    """
    import torch

    values = validate_motion_options(
        {
            "velocity_scale_m_s": velocity_scale_m_s,
            "acceleration_scale_m_s2": acceleration_scale_m_s2,
            "jerk_scale_m_s3": jerk_scale_m_s3,
            "peak_weight": peak_weight,
        }
    )
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    for value in (current, reference):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 3
            or value.shape[-1] != 3
            or value.shape[1] < 1
        ):
            raise ValueError("Motion inputs must be nonempty floating (T, P, 3)")
    if (
        current.shape != reference.shape
        or current.dtype != reference.dtype
        or current.device != reference.device
    ):
        raise ValueError("Motion inputs must share shape, dtype and device")
    delta = current - reference.detach()
    loss = current[:0].sum()
    for scale in (
        values["velocity_scale_m_s"],
        values["acceleration_scale_m_s2"],
        values["jerk_scale_m_s3"],
    ):
        if delta.shape[0] < 2:
            break
        delta = torch.diff(delta, dim=0) * fps
        squared = (delta / scale).square().sum(dim=-1)
        loss = loss + squared.mean() + values["peak_weight"] * squared.max()
    return loss
