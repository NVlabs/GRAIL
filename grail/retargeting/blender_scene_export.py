"""Export explicitly selected original static Blender meshes in GRAIL world metres.

Run with ``blender -b source.blend --disable-autoexec --python-exit-code 1
--python blender_scene_export.py -- --request request.json``. The request contains
``scene_blend: {path, sha256}``, exact ``object_names`` and a new ``output_npz``.
This reads the loaded scene without saving it or evaluating modified geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _check_static(obj):
    """Reject time-dependent geometry and transforms, including inherited motion."""
    current = obj
    while current is not None:
        if current.animation_data is not None:
            raise ValueError(f"Animated object or ancestor is unsupported: {current.name}")
        if current.constraints or current.modifiers:
            raise ValueError(f"Constraints/modifiers are unsupported: {current.name}")
        if current.rigid_body is not None or current.rigid_body_constraint is not None:
            raise ValueError(f"Rigid-body evaluation is unsupported: {current.name}")
        if current.parent is not None and current.parent_type != "OBJECT":
            raise ValueError(f"Only static object parenting is supported: {current.name}")
        current = current.parent
    if obj.data.animation_data is not None or obj.data.shape_keys is not None:
        raise ValueError(f"Animated mesh data or shape keys are unsupported: {obj.name}")


def export_static_scene(request):
    """Write original vertices transformed once by each selected matrix_world."""
    import bpy

    if not isinstance(request, dict) or set(request) != {
        "scene_blend",
        "object_names",
        "output_npz",
    }:
        raise ValueError("Require exactly scene_blend, object_names and output_npz")
    source = request["scene_blend"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueError("scene_blend must bind path and sha256")
    binding = _binding(source["path"])
    if binding["sha256"] != source["sha256"]:
        raise ValueError("Source Blender SHA-256 does not match the request")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != Path(binding["path"]):
        raise ValueError("Loaded Blender scene differs from the explicitly bound source")
    if bpy.data.is_dirty:
        raise ValueError("Export requires a freshly loaded, unedited source Blender scene")
    unit_scale = bpy.context.scene.unit_settings.scale_length
    if not np.isfinite(unit_scale) or unit_scale != 1.0:
        raise ValueError(
            "Source Blender scene must use scale_length=1 for GRAIL world metres; "
            "non-unit scene scales require an explicit upstream coordinate conversion"
        )
    names = request["object_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("object_names must contain unique, nonempty exact object names")
    output = Path(request["output_npz"]).expanduser().absolute()
    if output.suffix != ".npz":
        raise ValueError("output_npz must use the .npz suffix")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new static scene output: {output}")

    arrays, entries = {}, []
    for index, name in enumerate(names):
        obj = bpy.context.scene.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise ValueError(f"Selected source scene mesh is missing: {name}")
        _check_static(obj)
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
        ):
            raise ValueError(f"Source mesh has an invalid world transform: {name}")
        mesh = obj.data
        mesh.calc_loop_triangles()
        vertices = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        faces = np.asarray(
            [triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64
        )
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(f"Source mesh must contain finite vertices and triangles: {name}")
        vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(vertices).all():
            raise ValueError(f"World-space vertices are not finite: {name}")
        vertex_key, face_key = (
            f"static_scene_{index:03d}_vertices",
            f"static_scene_{index:03d}_faces",
        )
        arrays[vertex_key], arrays[face_key] = vertices, faces
        entries.append(
            {
                "name": name,
                "role": "support",
                "vertices_key": vertex_key,
                "faces_key": face_key,
                "source": binding,
                "source_object": name,
                "source_matrix_world": matrix.tolist(),
                "source_vertex_count": len(mesh.vertices),
                "source_polygon_count": len(mesh.polygons),
                "triangle_count": len(faces),
                "world_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            }
        )
    manifest = {
        "schema": "grail.static_scene.v1",
        "world_frame": "grail_z_up_m",
        "status": "PRESENT",
        "meshes": entries,
        "provenance": {
            "scene_blend": binding,
            "exporter": _binding(__file__),
            "blender_version": bpy.app.version_string,
            "source_frame": bpy.context.scene.frame_current,
            "source_unit_scale_length": unit_scale,
            "geometry_policy": "Original mesh vertices and loop triangles; matrix_world baked once",
        },
    }
    arrays["static_scene_json"] = np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False))
    if _binding(binding["path"]) != binding:
        raise ValueError("Source Blender file changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"output": _binding(output), "static_scene": manifest}


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text())
    print(json.dumps(export_static_scene(request), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
