"""Validate a generated SOMA object trajectory against its immutable reference.

A moving-source prefix may share any optimized pose within configured bounds;
its samples must remain exactly equal to one another. A wholly static original
is different: every original pose must pass through unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .soma_object import audit_object_trajectory, resolve_object_policy, validate_object_config


def validate_object_result(path, *, reference_path, reference_sha256, config, frame_count, fps):
    reference_path = Path(reference_path).resolve(strict=True)
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_sha256:
        raise ValueError("Original object trajectory changed")
    config = validate_object_config(config)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    required = {"obj_R", "obj_t", "reference_obj_R", "reference_obj_t", "object_scale",
                "timestamps", "fps", "world_frame", "reference_sha256", "config_json"}
    if set(arrays) != required:
        raise ValueError("Invalid optimized object output fields")
    if (str(arrays["world_frame"].item()) != "soma_retargeter_z_up_m"
            or str(arrays["reference_sha256"].item()) != reference_sha256
            or json.loads(str(arrays["config_json"].item())) != config
            or float(arrays["fps"]) != fps
            or arrays["timestamps"].shape != (frame_count,)
            or not np.allclose(arrays["timestamps"], np.arange(frame_count)/fps, atol=1e-7, rtol=0)):
        raise ValueError("Optimized object timing, reference or configuration binding differs")
    with np.load(reference_path, allow_pickle=False) as original:
        if (not np.array_equal(arrays["reference_obj_R"], original["obj_R"])
                or not np.array_equal(arrays["reference_obj_t"], original["obj_t"])
                or not np.array_equal(arrays["object_scale"], np.broadcast_to(original["object_scale"], (3,)))
                or not np.array_equal(arrays["timestamps"], original["timestamps"])):
            raise ValueError("Optimized object original-reference arrays changed")
    if arrays["obj_t"].shape != (frame_count, 3) or arrays["obj_R"].shape != (frame_count, 3, 3):
        raise ValueError("Optimized object frame count differs")
    policy = resolve_object_policy(arrays["reference_obj_R"], arrays["reference_obj_t"], config)
    if not policy["optimization_enabled"] and (
            not np.array_equal(arrays["obj_R"], arrays["reference_obj_R"])
            or not np.array_equal(arrays["obj_t"], arrays["reference_obj_t"])):
        raise ValueError("Optimized object output changed a static original trajectory")
    report = audit_object_trajectory(arrays["reference_obj_R"], arrays["reference_obj_t"],
        arrays["obj_R"], arrays["obj_t"], fps, config)
    report["motion_policy"] = policy
    if report["passed"] is not True:
        raise ValueError("Optimized object violates stationarity or original-reference bounds")
    return arrays, report
