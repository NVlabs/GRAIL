#!/usr/bin/env python3
"""Build an offline native-mesh phase contact-map viewer (no fitting).

Geometry uses the GRAIL export convention: world vertices, PyTorch3D row-vector
world-to-camera R/T, scalar focal length in pixels, image_size=[height,width].
Only load --motion from a trusted native checkpoint (joblib/pickle).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
from pathlib import Path

import numpy as np

from grail.optimization.contact_maps import PhaseContactMap
from grail.optimization.fixed_contact import file_sha256


def pack(array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        gzip=base64.b64encode(gzip.compress(array.tobytes(), mtime=0)).decode(),
    )


def normals(vertices, faces):
    result = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(result, faces[:, corner], normal)
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    return np.rint(result * 127).astype(np.int8)


def camera_to_viewer(rotation, translation):
    """PyTorch3D world->camera row vectors to Blender camera->world columns."""
    return rotation @ np.diag([-1.0, 1.0, -1.0]), -rotation @ translation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "map-spec",
        "binding",
        "binding-geometry",
        "geometry",
        "motion",
        "rest",
        "video",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--title", default="Dense phase contact maps")
    parser.add_argument("--default-frame", type=int, default=26)
    parser.add_argument("--view-distance", type=float, default=0.75)
    parser.add_argument("--crop", type=int, nargs=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import joblib

    geometry = np.load(args.geometry, allow_pickle=False)
    original = np.load(args.binding_geometry, allow_pickle=False)
    rest = np.load(args.rest, allow_pickle=False)
    binding = json.loads(args.binding.read_text())
    maps = PhaseContactMap.from_file(
        args.map_spec,
        binding=binding,
        human_faces=original["human_faces"],
        object_vertices=original["object_vertices"],
        object_faces=original["object_faces"],
        source_video_path=args.video,
        hand_vertex_ids={"left_hand": rest["L_Hand"], "right_hand": rest["R_Hand"]},
    )
    fields = maps.compact_fields()
    spec = maps.export_spec()
    hv, ov = geometry["human_vertices"], geometry["object_vertices"]
    frames = binding["frame_count"]
    if hv.shape != (frames, binding["human_vertex_count"], 3) or ov.shape != (
        frames,
        len(original["object_vertices"]),
        3,
    ):
        raise ValueError("Animated geometry shape does not match map binding")
    if not all(np.isfinite(a).all() for a in (hv, ov, rest["vertices"])):
        raise ValueError("Nonfinite geometry")
    for name in ("human_faces", "object_faces"):
        if not np.array_equal(geometry[name], original[name]):
            raise ValueError("Animated topology differs from bound original: " + name)
    if not np.array_equal(rest["faces"], original["human_faces"]):
        raise ValueError("Rest anatomy topology differs from native geometry")
    motion = joblib.load(args.motion)
    rotation, translation = [np.asarray(motion["obj_data"][key]) for key in ("obj_R", "obj_t")]
    reconstructed = (
        np.einsum("fij,vj->fvi", rotation, original["object_vertices"]) + translation[:, None]
    )
    residual = float(np.max(np.abs(reconstructed - ov)))
    if residual > 2e-5:
        raise ValueError(f"Object local/scale/transform mismatch: {residual} m")
    if not 0 <= args.default_frame < frames:
        raise ValueError("Default frame outside source timeline")
    center = ov[args.default_frame].mean(0)
    centered = hv.astype(np.float64) - center.astype(np.float64)
    step = max(0.00005, float(np.abs(centered).max()) / 32766)
    hp = np.rint(centered / step).astype(np.int16)
    hn = np.stack([normals(v, geometry["human_faces"]) for v in hv])
    groups = {}
    for side in ("left", "right"):
        for surface, mesh in (("hand", "human"), ("object", "object")):
            name = f"{side}_{surface}"
            groups[name] = dict(
                ids=pack(fields[name + "_vertex_ids"], "<u4"),
                confidence=pack(fields[name + "_strength"], "<f4"),
                dynamicMesh=mesh,
                uniquePositions=len(
                    np.unique(
                        np.round(
                            (rest["vertices"] if mesh == "human" else original["object_vertices"])[
                                fields[name + "_vertex_ids"]
                            ],
                            9,
                        ),
                        axis=0,
                    )
                ),
            )
    hands = {}
    for side, name in (("left", "L_Hand"), ("right", "R_Hand")):
        ids = rest[name].astype(np.int64)
        inverse = np.full(hv.shape[1], -1)
        inverse[ids] = np.arange(len(ids))
        faces = inverse[geometry["human_faces"]]
        faces = faces[(faces >= 0).all(axis=1)]
        hands[side] = dict(
            ids=pack(ids, "<u4"),
            positions=pack(rest["vertices"][ids], "<f4"),
            faces=pack(faces, "<u4"),
        )
    camera_r = geometry["camera_R"].reshape(3, 3)
    camera_t = geometry["camera_T"].reshape(3)
    camera_r, camera_t = camera_to_viewer(camera_r, camera_t)
    height, width = map(int, geometry["image_size"])
    crop = args.crop or [0, 0, width, height]
    if not (0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height):
        raise ValueError("Crop outside source image")
    paths = {
        name: getattr(args, name)
        for name in (
            "map_spec",
            "binding",
            "binding_geometry",
            "geometry",
            "motion",
            "rest",
            "video",
        )
    }
    provenance = dict(maps.provenance)
    provenance.update(
        prediction_sha256=file_sha256(args.map_spec),
        inputs={k: dict(path=str(v.resolve()), sha256=file_sha256(v)) for k, v in paths.items()},
        object_rigid_reconstruction_max_abs_m=residual,
        render_human_quantization_max_abs_m=float(
            np.abs(hp.astype(float) * step + center - hv).max()
        ),
    )
    data = dict(
        frameCount=frames,
        fps=binding["fps"],
        defaultFrame=args.default_frame,
        viewDistance=args.view_distance,
        step=step,
        center=center.tolist(),
        humanPositions=pack(hp, "<i2"),
        humanNormals=pack(hn, "|i1"),
        humanFaces=pack(geometry["human_faces"], "<u4"),
        objectLocal=pack(original["object_vertices"], "<f4"),
        objectNormals=pack(normals(original["object_vertices"], original["object_faces"]), "|i1"),
        objectFaces=pack(original["object_faces"], "<u4"),
        objectAffine=pack(np.concatenate([rotation, translation[:, :, None]], axis=2), "<f4"),
        cameraR=camera_r.tolist(),
        cameraT=camera_t.tolist(),
        focal=float(geometry["focal"]),
        resolution=[width, height],
        crop=crop,
        groups=groups,
        restHands=hands,
        phases=spec["hands"],
        evidence=dict(boundaries=[]),
        provenance=provenance,
    )
    template = Path(__file__).resolve().parents[2] / "grail/visualization/contact_map_template.html"
    text = template.read_text()
    replacements = {
        "__TITLE__": html.escape(args.title),
        "__STATUS__": "PROPOSAL · geometry unchanged",
        "__ROUNDING_MM__": f"{step*500:.4f}",
        "__LAST_FRAME__": str(frames - 1),
        "__DEFAULT_FRAME__": str(args.default_frame),
        "__VIDEO_BASE64__": base64.b64encode(args.video.read_bytes()).decode(),
        "__CONTACT_DATA__": json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
            "<", "\\u003c"
        ),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    receipt = dict(
        provenance,
        html_sha256=file_sha256(args.output),
        output=str(args.output.resolve()),
        code={str(p): file_sha256(p) for p in (Path(__file__), template)},
    )
    args.output.with_suffix(".provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                bytes=args.output.stat().st_size,
                optimizer_ready=False,
                geometry_changed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
