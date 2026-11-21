"""Bind native HOI object motion to an explicitly verified SOMA BVH export.

This copies the original object's material transform and scale; it does not
estimate contact, smooth motion, or manufacture an object from robot motion.
Pickle inputs must be trusted local GRAIL outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _object_motion(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            result = {name: data[name] for name in ("obj_R", "obj_t", "obj_scale", "fps")}
    else:
        import joblib

        data = joblib.load(path)
        human, obj = data["human_data"], data["obj_data"]
        fps = human.get("mocap_frame_rate", human.get("fps", data.get("fps")))
        if fps is None:
            raise ValueError("HOI source must declare its frame rate")
        result = {name: obj[name] for name in ("obj_R", "obj_t", "obj_scale")}
        result["fps"] = fps
    return {key: np.asarray(value) for key, value in result.items()}


def _rotation(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite 3x3 rotations")
    if not np.allclose(
        value @ value.swapaxes(-1, -2), np.eye(3), atol=2e-5, rtol=0
    ) or not np.allclose(np.linalg.det(value), 1, atol=2e-5, rtol=0):
        raise ValueError(f"{name} must contain proper rotations")
    return value


def export_object_sidecar(
    hoi_path,
    output_path,
    *,
    bvh_path,
    export_manifest,
    object_mesh_path,
    object_scale=None,
    scene_sidecar=None,
):
    """Write an immutable, timing-aligned sidecar for the official native converter.

    ``object_scale`` explicitly overrides the source scale only for a separately
    declared mesh representation (e.g. a mesh with scale baked in). Both values
    and the override are recorded; no mesh scale is inferred from a filename.
    """
    source = _binding(hoi_path)
    bvh = _binding(bvh_path)
    manifest_binding = _binding(export_manifest)
    mesh = _binding(object_mesh_path)
    manifest = json.loads(Path(manifest_binding["path"]).read_text())
    if (
        manifest.get("schema") != "grail.soma03.bvh_export.v1"
        or manifest.get("status") != "EXPORTED_UNREVIEWED"
    ):
        raise ValueError("Require a successful native SOMA03 BVH export manifest")
    if manifest.get("source") != source or manifest.get("bvh") != bvh:
        raise ValueError("Source/BVH bindings differ from the export manifest")
    if (
        manifest.get("source_world_frame") != "grail_z_up_m"
        or manifest.get("source_facing_direction") != "Mujoco"
    ):
        raise ValueError("Unsupported source/world convention")
    world = _rotation(manifest["world_from_grail"], "world_from_grail")
    if world.shape != (3, 3):
        raise ValueError("world_from_grail must be one rotation")
    offset = np.asarray(manifest.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("world_offset must be one finite XYZ translation")
    motion = _object_motion(Path(source["path"]))
    rotation = _rotation(motion["obj_R"], "obj_R")
    translation = np.asarray(motion["obj_t"], dtype=float)
    frames, fps = manifest["frame_count"], float(manifest["fps"])
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("Invalid frame count")
    if (
        not np.isfinite(fps)
        or fps <= 0
        or motion["fps"].size != 1
        or float(motion["fps"].reshape(())) != fps
    ):
        raise ValueError("Object FPS differs from the exported human timeline")
    if (
        rotation.shape != (frames, 3, 3)
        or translation.shape != (frames, 3)
        or not np.isfinite(translation).all()
    ):
        raise ValueError("Object transforms must cover the exact finite human timeline")
    scale = np.asarray(motion["obj_scale"] if object_scale is None else object_scale, dtype=float)
    # Legacy HOI pickles may represent one scalar or XYZ scale as a row/column.
    # Normalize these shapes only; never collapse an animated scale sequence.
    if scale.shape in ((1, 1), (3, 1), (1, 3)):
        scale = scale.reshape(-1)
    if scale.shape not in ((), (1,), (3,)) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(
            "Object scale must be positive scalar or XYZ; animated scaling is unsupported"
        )
    from .scene import static_scene_from_hoi

    scene = static_scene_from_hoi(
        hoi_path,
        scene_sidecar=scene_sidecar,
        world_from_grail=world,
        world_offset=offset,
        world_frame="soma_retargeter_z_up",
    )
    output = Path(output_path).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Use a new object handoff path: {output}")
    report = {
        "schema": "grail.retargeting.object_handoff.v1",
        "status": "UNREVIEWED",
        "source": source,
        "bvh": bvh,
        "export_manifest": manifest_binding,
        "object_mesh": mesh,
        "exporter": _binding(__file__),
        "frame_count": frames,
        "fps": fps,
        "world_frame": "soma_retargeter_z_up",
        "world_from_grail": world.tolist(),
        "world_offset": offset.tolist(),
        "grounding": manifest.get("grounding", {"enabled": False}),
        "source_object_scale": motion["obj_scale"].tolist(),
        "object_scale": scale.tolist(),
        "scale_override": object_scale is not None,
        "static_scene": json.loads(str(scene["static_scene_json"].item())),
    }
    for binding in (source, bvh, manifest_binding, mesh):
        if _binding(binding["path"]) != binding:
            raise ValueError("Bound handoff input changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            obj_R=world @ rotation,
            obj_t=translation @ world.T + offset,
            object_scale=scale,
            fps=np.asarray(fps),
            timestamps=np.arange(frames, dtype=float) / fps,
            world_frame=np.asarray(report["world_frame"]),
            source_sha256=np.asarray(bvh["sha256"]),
            object_mesh_path=np.asarray(mesh["path"]),
            object_mesh_sha256=np.asarray(mesh["sha256"]),
            hoi_source_sha256=np.asarray(source["sha256"]),
            export_manifest_sha256=np.asarray(manifest_binding["sha256"]),
            **scene,
        )
    report["output"] = _binding(output)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request", required=True, help="JSON with export_object_sidecar keyword arguments"
    )
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text())
    print(json.dumps(export_object_sidecar(**request), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
