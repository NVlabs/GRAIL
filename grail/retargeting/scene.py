"""Preserve original static scene geometry across human/robot model boundaries.

Geometry is packed in the robot world frame, in metres. Missing historical scene
metadata is unknown, not evidence for an empty scene. No category, object height
or robot pose is used to manufacture a table.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .handoff import _binding, _rotation

SCHEMA = "grail.static_scene.v1"
ROOT = Path(__file__).resolve().parents[2]


def _geometry(vertices, faces):
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("Static scene vertices must be nonempty Nx3")
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite static scene vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("Static scene faces must be nonempty triangular indices")
    # Old tensor loaders sometimes cast indices to float. Accept only exact integers.
    if faces.dtype.kind not in "iuf" or not np.isfinite(faces).all():
        raise ValueError("Invalid static scene face indices")
    if np.any(faces != np.floor(faces)) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Static scene face indices out of range or nonintegral")
    return vertices, faces.astype(np.int64)


def _packed(path):
    with np.load(path, allow_pickle=False) as data:
        if "static_scene_json" not in data:
            return None
        result = {key: data[key].copy() for key in data if key.startswith("static_scene_")}
    manifest = json.loads(str(result["static_scene_json"].item()))
    if manifest.get("schema") != SCHEMA or manifest.get("world_frame") != "grail_z_up_m":
        raise ValueError("Static scene input must be bound to grail_z_up_m")
    status, meshes = manifest.get("status"), manifest.get("meshes")
    if status not in {"PRESENT", "EMPTY", "UNKNOWN"} or not isinstance(meshes, list):
        raise ValueError("Invalid static scene status/meshes")
    if bool(meshes) != (status == "PRESENT"):
        raise ValueError("Static scene status contradicts its geometry")
    expected, names = {"static_scene_json"}, set()
    for index, item in enumerate(meshes):
        vk, fk = f"static_scene_{index:03d}_vertices", f"static_scene_{index:03d}_faces"
        if item.get("vertices_key") != vk or item.get("faces_key") != fk:
            raise ValueError("Invalid static scene geometry keys")
        if not item.get("name") or item["name"] in names:
            raise ValueError("Static scene names must be nonempty and unique")
        names.add(item["name"])
        result[vk], result[fk] = _geometry(result[vk], result[fk])
        expected.update((vk, fk))
    if set(result) != expected:
        raise ValueError("Unbound static scene arrays")
    return manifest, result


def _entry_geometry(entry, base):
    if "vertices" in entry or "faces" in entry:
        vertices, faces = _geometry(entry["vertices"], entry["faces"])
        binding = entry.get("source_blend")
    elif "path" in entry:
        import trimesh

        path = Path(entry["path"]).expanduser()
        if not path.is_absolute():
            candidates = [base / path]
            # Historical generation records repository-relative data/Scene paths.
            # Restrict this compatibility rule to that actual recorded contract.
            if path.parts[:2] == ("data", "Scene"):
                candidates.append(ROOT / path)
            found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
            if len(found) > 1 and len({_binding(p)["sha256"] for p in found}) > 1:
                raise ValueError("Ambiguous relative static scene mesh path")
            path = found[0] if found else candidates[0]
        binding = _binding(path)
        mesh = trimesh.load(binding["path"], process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices, faces = _geometry(mesh.vertices, mesh.faces)
    elif "size" in entry:
        import trimesh

        size = np.asarray(entry["size"], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Authored table size must be positive XYZ")
        mesh = trimesh.creation.box(extents=size)
        vertices, faces = _geometry(mesh.vertices, mesh.faces)
        binding = None  # The authored box is bound to the source checkpoint below.
    else:
        raise ValueError("Static scene entry requires original mesh or explicit box geometry")
    rotation = _rotation(entry.get("rot", np.eye(3)), "static scene rotation")
    scale = np.asarray(entry.get("scale", np.ones(3)), dtype=float).reshape(-1)
    position = np.asarray(entry.get("pos", np.zeros(3)), dtype=float).reshape(-1)
    if rotation.shape != (3, 3) or scale.size not in (1, 3) or position.shape != (3,):
        raise ValueError("Invalid static scene transform")
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(position).all():
        raise ValueError("Static scene transforms must be finite with positive scale")
    return (vertices * scale) @ rotation.T + position, faces, binding


def static_scene_from_hoi(
    hoi_path,
    *,
    scene_sidecar=None,
    world_from_grail=None,
    world_scale=1.0,
    world_offset=(0.0, 0.0, 0.0),
    world_frame="grail_z_up_m",
):
    """Use native scene_data, or an explicit original-scene sidecar for legacy data.

    ``scene_sidecar`` is an explicit authored override, recorded in provenance;
    it must contain original geometry in GRAIL world coordinates. Coordinate
    conversion must be identical to the moving object's conversion, applied once.
    """
    source = _binding(hoi_path)
    rotation = _rotation(
        np.eye(3) if world_from_grail is None else world_from_grail, "world_from_grail"
    )
    offset = np.asarray(world_offset, dtype=float)
    if rotation.shape != (3, 3) or offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid static scene world transform")
    if not np.isfinite(world_scale) or world_scale <= 0 or not world_frame:
        raise ValueError("Invalid static scene scale/world frame")
    manifest = dict(schema=SCHEMA, world_frame="grail_z_up_m", status="UNKNOWN", meshes=[])
    arrays, packed, sidecar_binding = {}, None, None
    if scene_sidecar is not None:
        sidecar_binding = _binding(scene_sidecar)
        packed = _packed(sidecar_binding["path"])
        if packed is None:
            raise ValueError("Explicit scene sidecar lacks static_scene_json")
    elif Path(source["path"]).suffix == ".npz":
        packed = _packed(source["path"])
    else:
        import joblib

        data = joblib.load(source["path"])
        scene = data.get("scene_data", data.get("scene", data.get("static_objects")))
        if scene:
            if not isinstance(scene, dict):
                raise ValueError("HOI scene_data must be a mapping")
            for index, (name, entry) in enumerate(sorted(scene.items())):
                vertices, faces, binding = _entry_geometry(entry, Path(source["path"]).parent)
                vk, fk = f"static_scene_{index:03d}_vertices", f"static_scene_{index:03d}_faces"
                arrays.update({vk: vertices, fk: faces})
                manifest["meshes"].append(
                    dict(
                        name=name,
                        role=entry.get("role", "static"),
                        vertices_key=vk,
                        faces_key=fk,
                        source=binding or source,
                        source_object=entry.get("source_object", name),
                    )
                )
            manifest["status"] = "PRESENT"
    if packed is not None:
        manifest, arrays = packed
        manifest = copy.deepcopy(manifest)
        arrays.pop("static_scene_json")
    for item in manifest["meshes"]:
        key = item["vertices_key"]
        arrays[key] = (arrays[key] * world_scale) @ rotation.T + offset
        item["world_bounds"] = [arrays[key].min(axis=0).tolist(), arrays[key].max(axis=0).tolist()]
    manifest["world_frame"] = world_frame
    manifest["provenance"] = dict(
        upstream=manifest.get("provenance", {}),
        hoi_source=source,
        scene_sidecar=sidecar_binding,
        world_from_grail=rotation.tolist(),
        world_scale=float(world_scale),
        world_offset=offset.tolist(),
        exporter=_binding(__file__),
    )
    for binding in (source, sidecar_binding):
        if binding is not None and _binding(binding["path"]) != binding:
            raise ValueError("Static scene input changed during export")
    arrays["static_scene_json"] = np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return arrays
