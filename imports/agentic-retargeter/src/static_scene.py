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


def assert_static_scene_equal(template, candidate):
    """A render candidate may change qpos, never its frozen static scene."""
    _, expected = static_scene_payload(template)
    _, actual = static_scene_payload(candidate)
    if expected.keys() != actual.keys():
        raise ValueError("Candidate changes static scene fields")
    if expected and not np.array_equal(template["world_frame"], candidate["world_frame"]):
        raise ValueError("Candidate changes static scene world_frame")
    for key in expected:
        if expected[key].dtype != actual[key].dtype or not np.array_equal(expected[key], actual[key]):
            raise ValueError("Candidate changes static scene field: " + key)


def static_scene_fingerprint(manifest, arrays):
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _names(index):
    # Blender names have a byte limit; source names remain explicit metadata.
    return f"GRAIL_StaticScene_{index:03d}", f"GRAIL_StaticScene_{index:03d}_Mesh"


def _visible_layer(layer, collection):
    if layer.collection == collection:
        return not layer.exclude and not layer.hide_viewport
    return any(_visible_layer(child, collection) for child in layer.children)


def _check_blender_collection(bpy, collection, manifest, arrays, fingerprint):
    if collection.get(OWNER_KEY) != SCHEMA or collection.get(FINGERPRINT_KEY) != fingerprint:
        raise ValueError("Static scene collection collision or stale content fingerprint")
    if (collection.name not in bpy.context.scene.collection.children or collection.children
            or collection.hide_render or collection.hide_viewport
            or not _visible_layer(bpy.context.view_layer.layer_collection, collection)):
        raise ValueError("Static scene collection is moved, hidden or contains unexpected children")
    expected_names = {_names(i)[0] for i in range(len(manifest["meshes"]))}
    if set(collection.objects.keys()) != expected_names:
        raise ValueError("Static scene collection contains missing or unexpected objects")
    if {obj.name for obj in bpy.data.objects if obj.get(OWNER_KEY) == SCHEMA} != expected_names:
        raise ValueError("Static scene has unexpected managed objects outside its collection")
    for index, mesh in enumerate(manifest["meshes"]):
        name, data_name = _names(index)
        obj = bpy.data.objects[name]
        if (obj.get(OWNER_KEY) != SCHEMA or obj.get(FINGERPRINT_KEY) != fingerprint
                or obj.get("grail_static_scene_name") != mesh["name"]
                or obj.get("grail_static_scene_role") != mesh["role"]
                or obj.get("grail_static_scene_source_json") != json.dumps(mesh["source"], sort_keys=True)
                or obj.type != "MESH" or obj.data.name != data_name):
            raise ValueError("Static scene managed object collision or stale identity: " + name)
        if (obj.parent is not None or obj.constraints or obj.modifiers or obj.animation_data
                or obj.rigid_body or obj.data.animation_data or obj.data.shape_keys
                or len(obj.users_collection) != 1 or obj.users_collection[0] != collection
                or obj.data.users != 1 or obj.hide_render or obj.hide_viewport or obj.hide_get()
                or obj.display_type != "TEXTURED" or obj.data.materials
                or not obj.visible_camera or obj.is_holdout or obj.is_shadow_catcher
                or obj.instance_type != "NONE"
                or not np.array_equal(np.asarray(obj.matrix_world), np.eye(4))
                or not np.array_equal(np.asarray(obj.matrix_basis), np.eye(4))):
            raise ValueError("Static scene managed object has changed transforms, visibility or evaluation: " + name)
        vertices = np.asarray([tuple(vertex.co) for vertex in obj.data.vertices], dtype=np.float32)
        faces = [tuple(face.vertices) for face in obj.data.polygons]
        if (not np.array_equal(vertices, arrays[mesh["vertices_key"]].astype(np.float32))
                or any(len(face) != 3 for face in faces)
                or not np.array_equal(np.asarray(faces), arrays[mesh["faces_key"]])):
            raise ValueError("Static scene managed mesh was mutated: " + name)


def ensure_blender_static_scene(values, *, bpy_module=None):
    """Import once, then verify original world triangles on every preview.

    Managed content is never silently repaired or replaced. Legacy scenes with
    absent metadata remain UNKNOWN and receive no inferred environment mesh.
    """
    if bpy_module is None:
        import bpy as bpy_module
    bpy = bpy_module
    manifest, arrays = static_scene_payload(values)
    fingerprint = static_scene_fingerprint(manifest, arrays)
    collection = bpy.data.collections.get(COLLECTION)
    if collection is not None:
        _check_blender_collection(bpy, collection, manifest, arrays, fingerprint)
    elif any(obj.get(OWNER_KEY) == SCHEMA for obj in bpy.data.objects):
        raise ValueError("Static scene has orphaned managed objects without its collection")
    elif manifest["meshes"]:
        # Check every collision before creating any datablock.
        for index, mesh in enumerate(manifest["meshes"]):
            name, data_name = _names(index)
            if name in bpy.data.objects or data_name in bpy.data.meshes:
                raise ValueError("Static scene reserved Blender name collision: " + name)
            vertices = arrays[mesh["vertices_key"]]
            if np.any(np.abs(vertices) > np.finfo(np.float32).max):
                raise ValueError("Static scene world vertices exceed Blender coordinate precision")
        collection = bpy.data.collections.new(COLLECTION)
        collection[OWNER_KEY], collection[FINGERPRINT_KEY] = SCHEMA, fingerprint
        bpy.context.scene.collection.children.link(collection)
        for index, mesh in enumerate(manifest["meshes"]):
            name, data_name = _names(index)
            data = bpy.data.meshes.new(data_name)
            data.from_pydata(arrays[mesh["vertices_key"]].tolist(), [], arrays[mesh["faces_key"]].tolist())
            data.update()
            obj = bpy.data.objects.new(name, data)
            obj[OWNER_KEY], obj[FINGERPRINT_KEY] = SCHEMA, fingerprint
            obj["grail_static_scene_name"], obj["grail_static_scene_role"] = mesh["name"], mesh["role"]
            obj["grail_static_scene_source_json"] = json.dumps(mesh["source"], sort_keys=True)
            collection.objects.link(obj)
        bpy.context.view_layer.update()
        _check_blender_collection(bpy, collection, manifest, arrays, fingerprint)
    return dict(status=manifest["status"], sha256=fingerprint,
                meshes=len(manifest["meshes"]), collection=COLLECTION if collection is not None else None,
                collision_checked=False)
