"""Compile authored material contacts for the optional SOMA/Newton IK bridge.

No targets are inferred. Frames are zero-based, intervals are half-open, and all
object transforms are explicitly in the SOMA solver's metric Z-up world.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np

from .soma_object_policy import contact_loss_policy


def compile_contacts(path, *, frame_count, fps, model_xml, include_object_scene=False,
                     allow_empty_moving=False):
    """Compile material constraints, optionally serving a validated pose bridge.

    The adapter may set ``allow_empty_moving`` only after validating external
    native hand targets. It allows zero authored anchors without changing any
    existing contact/static-object policy or inventing diagnostic anchors.
    """
    if type(allow_empty_moving) is not bool:
        raise ValueError("allow_empty_moving must be boolean")
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    required = {"schema_version", "world_frame", "post_processing", "object_mesh_npz",
                "object_trajectory", "contacts"}
    if required - set(spec) or set(spec) - required - {"collision_avoidance"} or spec["schema_version"] != 1:
        raise ValueError(f"Contact spec requires {sorted(required)}; optional collision_avoidance")
    if spec["world_frame"] != "soma_retargeter_z_up_m":
        raise ValueError("Contact world_frame must be soma_retargeter_z_up_m")
    if spec["post_processing"] != "disabled":
        raise ValueError("Declare post_processing='disabled' to prevent later wrist overwrite")
    inputs = {"spec": path}
    for key in ("object_mesh_npz", "object_trajectory"):
        inputs[key] = (path.parent / spec[key]).resolve()
    with np.load(inputs["object_mesh_npz"], allow_pickle=False) as mesh:
        vertices = np.asarray(mesh["vertices"], dtype=float)
        faces = np.asarray(mesh["faces"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Invalid original triangle mesh")
    with np.load(inputs["object_trajectory"], allow_pickle=False) as motion:
        rotations = np.asarray(motion["obj_R"], dtype=float)
        positions = np.asarray(motion["obj_t"], dtype=float)
        scale = np.broadcast_to(np.asarray(motion["object_scale"], dtype=float), (3,))
        if not np.isclose(float(motion["fps"]), fps, rtol=0, atol=1e-8):
            raise ValueError("Object FPS differs from source BVH")
        times = np.asarray(motion["timestamps"], dtype=float)
    if (times.shape != (frame_count,) or not np.allclose(times, np.arange(frame_count)/fps, atol=1e-7, rtol=0)
            or rotations.shape != (frame_count, 3, 3) or positions.shape != (frame_count, 3)
            or not np.isfinite(rotations).all() or not np.isfinite(positions).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError("Object requires matching source timing and rigid metric world transforms")
    bodies = {b.get("name") for b in ET.parse(model_xml).iter("body")}
    policy = contact_loss_policy(dict(object_reference_rotations=rotations,
                                     object_reference_positions=positions))
    contacts = spec["contacts"]
    if not isinstance(contacts, list) or (not contacts and policy["enabled"] and not allow_empty_moving):
        raise ValueError("At least one explicitly authored contact is required for a moving object")
    keys = {"name", "body_name", "point_local", "normal_local", "object_face_id",
            "object_barycentric", "frames", "position_weight"}
    optional = {"object_normal_sign", "ramp_frames", "normal_weight"}
    names, points, normals, target_points, target_normals, gains, weights, body_names = ([] for _ in range(8))
    object_points, object_normals = [], []
    occupied = set()
    for item in contacts:
        if not isinstance(item, dict) or keys - set(item) or set(item) - keys - optional:
            raise ValueError("Invalid contact fields")
        name, body = item["name"], item["body_name"]
        if not isinstance(name, str) or not name or name in names or body not in bodies:
            raise ValueError("Contact names must be unique and bodies must exist in original model")
        def vector(key):
            value = np.asarray(item[key], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key}")
            return value
        point, normal, bary = (vector(key) for key in ("point_local", "normal_local", "object_barycentric"))
        if not np.isclose(np.linalg.norm(normal), 1, atol=1e-6):
            raise ValueError("Robot pad normal must be unit length")
        if (bary < 0).any() or not np.isclose(bary.sum(), 1, atol=1e-8):
            raise ValueError("Invalid object barycentric coordinates")
        face = item["object_face_id"]
        if type(face) is not int or not 0 <= face < len(faces):
            raise ValueError("Original object face is out of range")
        interval = item["frames"]
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(f) is not int for f in interval) or not 0 <= interval[0] < interval[1] <= frame_count):
            raise ValueError("Contact frames must be a source-indexed half-open interval")
        start, stop = interval
        # An anchor revision must be a new, non-overlapping phase for that pad.
        identities = {(body, tuple(point), f) for f in range(start, stop)}
        if occupied & identities:
            raise ValueError("Overlapping material targets on the same pad")
        occupied.update(identities)
        sign = item.get("object_normal_sign", 1)
        ramp = item.get("ramp_frames", 0)
        pw, nw = item["position_weight"], item.get("normal_weight", 0.0)
        if (isinstance(sign, bool) or sign not in (-1, 1) or type(ramp) is not int
                or ramp < 0 or ramp * 2 > stop-start
                or any(type(w) not in (int, float) or not np.isfinite(w) or w < 0 for w in (pw, nw)) or pw == 0):
            raise ValueError("Invalid contact weights, normal sign or ramp")
        triangle = vertices[faces[face]] * scale
        obj_normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        length = np.linalg.norm(obj_normal)
        if length < 1e-12:
            raise ValueError("Degenerate contact triangle")
        obj_normal *= sign / length
        gain = np.zeros(frame_count)
        gain[start:stop] = 1
        if ramp:
            # Endpoint-inclusive raised cosine; never activates outside the phase.
            blend = .5 - .5*np.cos(np.pi*np.arange(1, ramp+1)/(ramp+1))
            gain[start:start+ramp] = blend
            gain[stop-ramp:stop] = blend[::-1]
        names.append(name); body_names.append(body); points.append(point); normals.append(normal)
        object_points.append(bary @ triangle); object_normals.append(obj_normal)
        target_points.append(np.einsum("tij,j->ti", rotations, bary @ triangle) + positions)
        target_normals.append(np.einsum("tij,j->ti", rotations, obj_normal))
        gains.append(gain); weights.append([pw, nw])
    arrays = dict(names=np.asarray(names, dtype=str), body_names=np.asarray(body_names, dtype=str),
                  points=np.asarray(points).reshape(-1, 3), normals=np.asarray(normals).reshape(-1, 3),
                  targets=np.stack(target_points, axis=1) if names else np.empty((frame_count, 0, 3)),
                  target_normals=np.stack(target_normals, axis=1) if names else np.empty((frame_count, 0, 3)),
                  gains=np.stack(gains, axis=1) if names else np.empty((frame_count, 0)),
                  weights=np.asarray(weights).reshape(-1, 2), fps=np.asarray(fps),
                  object_local_points=np.asarray(object_points).reshape(-1, 3),
                  object_local_normals=np.asarray(object_normals).reshape(-1, 3),
                  object_reference_rotations=rotations.copy(), object_reference_positions=positions.copy(),
                  object_scale=scale.copy(), object_timestamps=times.copy())
    summary = {"status": "AUTHORED_CONTACTS_UNREVIEWED", "frame_count": frame_count,
               "contact_count": len(names), "names": names, "post_processing": "disabled",
               "contact_loss_policy": policy,
               "position_objective_count": len(names) if policy["enabled"] else 0,
               "normal_objective_count": int(np.count_nonzero(arrays["weights"][:, 1])) if policy["enabled"] else 0,
               "normal_alignment_contacts": [name for name, weight in zip(names, weights) if weight[1] > 0],
               "objective": ("sum(gain * (position_weight * gap_m^2 + normal_weight * ||pad_normal + object_normal||^2))"
                             if policy["enabled"] else "disabled for static source object; authored contacts are diagnostic only"),
               "active_frames_per_contact": [int(np.count_nonzero(g)) for g in gains]}
    if "collision_avoidance" in spec:
        from .soma_collision import compile_collision
        collision_arrays, collision_summary = compile_collision(spec["collision_avoidance"], model_xml,
            vertices*scale, faces, rotations, positions, config_dir=path.parent,
            include_object_scene=include_object_scene)
        arrays.update(collision_arrays)
        summary["collision_avoidance"] = collision_summary
        if "native_npz" in collision_summary["static_scene"]:
            inputs["static_scene_npz"] = Path(collision_summary["static_scene"]["native_npz"])
    return arrays, inputs, summary
