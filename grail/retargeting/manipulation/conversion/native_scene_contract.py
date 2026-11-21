"""Body-independent static geometry carried in a native HOI NPZ.

Vertices are already in the native world frame, in metres. This module never
estimates a table, applies human/robot scale, or supplies collision evidence.
Source paths are provenance; the packed triangles are the portable geometry.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np


SCHEMA = "grail.static_scene.v1"
MANIFEST_KEY = "static_scene_json"
COLLECTION = "GRAIL_StaticScene"
OWNER_KEY = "grail_static_scene_schema"
FINGERPRINT_KEY = "grail_static_scene_sha256"


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    return value


def _scalar_string(value, label):
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(label + " must be a scalar string")
    text = value.item()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _string(text, label)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate static scene JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite static scene JSON value: " + value)


def static_scene_payload(values, *, world_frame=None):
    """Validate and return (manifest, exact arrays) without modifying input.

    Legacy absence is UNKNOWN and adds no arrays. The reserved namespace is
    closed: orphan fields, aliases of native arrays, and reused keys fail closed.
    """
    fields = {key for key in values if key.startswith("static_scene_")}
    if MANIFEST_KEY not in values:
        if fields:
            raise ValueError("Static scene arrays require static_scene_json")
        return dict(schema=SCHEMA, status="UNKNOWN", meshes=[],
                    world_frame=world_frame, provenance={"reason": "not_provided"}), {}
    if "world_frame" not in values:
        raise ValueError("Static scene requires the native world_frame")
    native_world = _scalar_string(values["world_frame"], "native world_frame")
    if world_frame is not None and native_world != world_frame:
        raise ValueError("Static scene native world_frame differs from the expected world_frame")
    manifest = json.loads(_scalar_string(values[MANIFEST_KEY], MANIFEST_KEY),
                          object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("Expected static scene schema " + SCHEMA)
    # Also reject overflowed JSON numbers (e.g. 1e999) in provenance/extra fields.
    json.dumps(manifest, allow_nan=False)
    if manifest.get("world_frame") != native_world:
        raise ValueError("Static scene world_frame differs from native world_frame")
    status = manifest.get("status")
    if status not in ("PRESENT", "EMPTY", "UNKNOWN"):
        raise ValueError("Static scene status must be PRESENT, EMPTY or UNKNOWN")
    meshes = manifest.get("meshes")
    if not isinstance(meshes, list) or (status == "PRESENT") != bool(meshes):
        raise ValueError("Static scene PRESENT requires meshes; EMPTY/UNKNOWN require no meshes")
    if not isinstance(manifest.get("provenance"), dict):
        raise ValueError("Static scene provenance must be an object")
    arrays = {MANIFEST_KEY: np.asarray(values[MANIFEST_KEY])}
    names = set()
    for index, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            raise ValueError("Static scene mesh must be an object")
        name = _string(mesh.get("name"), "Static scene mesh name")
        if name in names:
            raise ValueError("Static scene mesh names must be unique")
        names.add(name)
        _string(mesh.get("role"), "Static scene mesh role")
        source = mesh.get("source")
        if not isinstance(source, dict):
            raise ValueError("Static scene mesh requires source path and SHA-256")
        _string(source.get("path"), "Static scene source path")
        if not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
            raise ValueError("Static scene source sha256 must be a SHA-256 hex digest")
        for kind in ("vertices", "faces"):
            key = f"static_scene_{index:03d}_{kind}"
            if mesh.get(kind + "_key") != key or key not in values:
                raise ValueError("Static scene requires its reserved array key: " + key)
            arrays[key] = np.asarray(values[key])
        vertices, faces = arrays[mesh["vertices_key"]], arrays[mesh["faces_key"]]
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
                or vertices.dtype.kind != "f" or not np.isfinite(vertices).all()):
            raise ValueError("Static scene vertices must be finite floating Nx3 world coordinates")
        if (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
                or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError("Static scene faces must be integer Mx3 triangles with indices in range")
    if fields != arrays.keys():
        raise ValueError("Unexpected reserved static scene fields: " + ", ".join(sorted(fields - arrays.keys())))
    return manifest, arrays
