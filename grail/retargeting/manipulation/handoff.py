#!/usr/bin/env python3
"""Collect one retargeted motion into a tracking handoff directory.

Inputs are the native robot motion from agentic retargeting (``grail.retargeting.agentic``:
Unitree G1 + DEx3, 50 qpos / 43 named DOFs, object poses, scale and support scene) and the
reviewed HOI contact map of the same source video (``grail.contact_map.v1``)::

    python -m grail.retargeting.manipulation.handoff \\
        --native runs/case/retarget/native/robot_UNREVIEWED.npz \\
        --contact-map runs/case/recon/contact_map.json \\
        --motion-key table_046_s02 --output runs/case/tracking_handoff

The handoff copies the inputs unchanged, adds the object mesh as a vertices/faces NPZ in the
original OBJ order, checks the native and contact-map contracts, and records which support
context tracking uses (``table-block`` for a table scene, ``native`` for floor scenes). It
runs in any environment with NumPy; no retargeting, resampling or cropping is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from grail.retargeting.manipulation.paths import sha256

SCHEMA = "grail.tracking.manipulation.handoff.v1"


def ordered_obj(path):
    """Vertices and triangle faces exactly in OBJ ``v`` / ``f`` order (texture/normal indices dropped)."""
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Object OBJ must be triangulated")
            face = []
            for token in fields[1:]:
                index = int(token.split("/")[0])
                face.append(index - 1 if index > 0 else len(vertices) + index)
            faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def check_native(native) -> dict:
    qpos, fps = native["qpos"], float(native["fps"])
    frames = len(qpos)
    if qpos.shape != (frames, 50) or frames < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Native motion must be G1 + DEx3 with 50 qpos per frame")
    for key in ("qpos", "obj_R", "obj_t", "object_scale", "timestamps"):
        if not np.isfinite(native[key]).all():
            raise ValueError(f"Nonfinite native field: {key}")
    if not np.allclose(native["timestamps"], np.arange(frames) / fps, atol=1e-9):
        raise ValueError("Native timestamps must be the uniform source clock")
    columns = dict(zip(native["dof_names"].tolist(), native["qposadr"].tolist()))
    if len(columns) != 43 or set(columns.values()) != set(range(7, 50)):
        raise ValueError("Expected 43 named G1 + DEx3 DOFs at qpos addresses 7..49")
    if native["obj_R"].shape != (frames, 3, 3) or native["obj_t"].shape != (frames, 3):
        raise ValueError("Object trajectory must cover every native frame")
    if "static_scene_json" not in native:
        raise ValueError("Native motion must carry its support scene (static_scene_json)")
    scene = json.loads(str(native["static_scene_json"]))
    meshes = scene.get("meshes", [])
    # Older handoffs can contain unlabelled table geometry.
    has_raised_support = any(mesh.get("role") not in ("floor", "ground") for mesh in meshes)
    return {"frames": frames, "fps": fps, "support_meshes": len(meshes),
            "support": "table-block" if has_raised_support else "native"}


def check_contact_map(contact_map: dict, frames: int, fps: float) -> None:
    if contact_map.get("schema") != "grail.contact_map.v1":
        raise ValueError("Contact map must use schema grail.contact_map.v1")
    binding = contact_map.get("binding", {})
    if int(binding.get("frame_count", frames)) != frames or abs(float(binding.get("fps", fps)) - fps) > 1e-9:
        raise ValueError("Contact map clock differs from the native motion")
    for side in ("left", "right"):
        covered = np.zeros(frames, bool)
        for phase in contact_map.get("hands", {}).get(side + "_hand", []):
            first, end = map(int, phase["frames"])
            if first < 0 or end > frames or end <= first or covered[first:end].any():
                raise ValueError(f"Invalid or overlapping {side}-hand contact interval {first}:{end}")
            if phase["state"] not in ("contact", "no_contact", "unknown"):
                raise ValueError(f"Unknown contact state {phase['state']}")
            covered[first:end] = True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native", type=Path, required=True, help="Native G1 + DEx3 robot/object motion NPZ")
    p.add_argument("--contact-map", type=Path, required=True, help="Reviewed HOI contact map (grail.contact_map.v1)")
    p.add_argument("--object-mesh", type=Path, help="Original object OBJ (default: the path recorded in the native NPZ)")
    p.add_argument("--motion-key", required=True, help="Unique motion name, e.g. dataset__object_s01")
    p.add_argument("--support", choices=("auto", "table-block", "native"), default="auto",
                   help="Support context for tracking: table-block for table scenes, native for floor scenes")
    p.add_argument("--output", type=Path, required=True, help="New handoff directory")
    args = p.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new handoff directory: {output}")
    with np.load(args.native, allow_pickle=False) as data:
        native = {key: data[key] for key in data.files}
    summary = check_native(native)
    contact_map = json.loads(args.contact_map.read_text())
    check_contact_map(contact_map, summary["frames"], summary["fps"])
    mesh_path = args.object_mesh or Path(str(native["object_mesh_path"]))
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Object OBJ not found ({mesh_path}); pass --object-mesh")
    if "object_mesh_sha256" in native and sha256(mesh_path) != str(native["object_mesh_sha256"]):
        raise ValueError("Object OBJ bytes differ from the mesh recorded in the native motion")
    support = args.support
    if support == "auto":
        support = summary["support"]
    output.mkdir(parents=True)
    shutil.copyfile(args.native, output / "final_native.npz")
    shutil.copyfile(args.contact_map, output / "contact_map.json")
    shutil.copyfile(mesh_path, output / "object_mesh.obj")
    vertices, faces = ordered_obj(mesh_path)
    np.savez_compressed(output / "object_mesh.npz", vertices=vertices, faces=faces)
    record = {"schema": SCHEMA, "motion_key": args.motion_key, "support": support,
              "frames": summary["frames"], "fps": summary["fps"], "native_support_meshes": summary["support_meshes"],
              "created_at": datetime.now(timezone.utc).isoformat(),
              "sources": {"native": str(args.native.resolve()), "contact_map": str(args.contact_map.resolve()),
                          "object_mesh": str(mesh_path.resolve())},
              "files": {name: sha256(output / name) for name in
                        ("final_native.npz", "contact_map.json", "object_mesh.obj", "object_mesh.npz")}}
    (output / "handoff.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("motion_key", "support", "frames", "fps")}))


if __name__ == "__main__":
    main()
