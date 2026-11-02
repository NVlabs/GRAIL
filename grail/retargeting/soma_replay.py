"""Render a G1+DEx3 CSV together with its GRAIL object and a support table.

Run this module with ``imports/soma-retargeter/.venv/bin/python``.  The replay
is kinematic: it is intended to inspect retarget, object, and scene alignment,
not to certify dynamics, collision freedom, or grasp success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path


def _binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _load_scene(path, object_mesh=None):
    import numpy as np
    import trimesh

    source = Path(path).expanduser().resolve(strict=True)
    with source.open("rb") as stream:
        data = pickle.load(stream)
    required = {"obj_R", "obj_t", "obj_scale"}
    if not required.issubset(data["obj_data"]):
        raise ValueError(f"Object data must contain {sorted(required)}")
    rotations = np.asarray(data["obj_data"]["obj_R"], dtype=np.float32)
    translations = np.asarray(data["obj_data"]["obj_t"], dtype=np.float32)
    scale = np.asarray(data["obj_data"]["obj_scale"], dtype=np.float32).reshape(3)
    if rotations.shape != (len(translations), 3, 3) or translations.shape[1:] != (3,):
        raise ValueError("Object trajectory must be [T,3,3] rotations and [T,3] positions")
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("Object trajectory contains non-finite values")
    mesh_path = Path(object_mesh or data["object_path"]).expanduser().resolve(strict=True)
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Object mesh contains no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Replay requires a triangular object mesh")

    return {
        "source": source,
        "mesh_path": mesh_path,
        "vertices": vertices,
        "faces": faces,
        "rotations": rotations,
        "translations": translations,
        "scale": scale,
    }


def _synthetic_table(scene, table_size):
    """Build the public replay table directly below the initial object mesh."""
    import numpy as np
    import trimesh

    size = np.asarray(table_size, dtype=np.float32).reshape(3)
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Table size must contain three positive finite values")

    initial_vertices = (scene["vertices"] * scene["scale"]) @ scene["rotations"][0].T + scene[
        "translations"
    ][0]
    surface_z = float(initial_vertices[:, 2].min())
    center = np.asarray(
        [
            scene["translations"][0, 0],
            scene["translations"][0, 1],
            surface_z - size[2] / 2.0,
        ],
        dtype=np.float32,
    )

    tabletop = trimesh.creation.box(extents=size)
    tabletop.apply_translation(center)
    parts = [tabletop]

    # Match the simple four-leg table used by the internal replay renderer.
    leg_height = float(center[2] - size[2] / 2.0)
    leg_count = 0
    if leg_height > 0:
        leg_width = 0.04
        x_offset = max(0.0, float(size[0] / 2.0 - 0.03 - leg_width / 2.0))
        y_offset = max(0.0, float(size[1] / 2.0 - 0.10 - leg_width / 2.0))
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                leg = trimesh.creation.box(extents=(leg_width, leg_width, leg_height))
                leg.apply_translation(
                    (
                        center[0] + x_sign * x_offset,
                        center[1] + y_sign * y_offset,
                        leg_height / 2.0,
                    )
                )
                parts.append(leg)
                leg_count += 1

    mesh = trimesh.util.concatenate(parts)
    return {
        "pos": center,
        "size": size,
        "surface_z": surface_z,
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "faces": np.asarray(mesh.faces, dtype=np.int32),
        "leg_count": leg_count,
    }


def _csv_shape(path):
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        frames = sum(1 for _ in reader)
    if len(header) != 50 or len(header[7:]) != 43:
        raise ValueError("Scene replay requires a native 43-joint G1+DEx3 CSV")
    return frames, header[7:]


def render_scene(
    csv_path,
    hoi_path,
    target_path,
    output,
    *,
    soma_root="imports/soma-retargeter",
    object_mesh=None,
    table_size=(2.0, 0.6, 0.04),
    fps=24.0,
    width=1280,
    height=720,
    camera=(2.35, -2.65, 1.75),
    look_at=(0.1, 0.15, 0.78),
    device="cuda:0",
):
    """Render a frame-aligned kinematic replay and return its audit receipt."""
    import newton
    import numpy as np
    import warp as wp
    from soma_retargeter.io import csv as csv_io
    from soma_retargeter.pipelines import utils as pipeline_utils
    from soma_retargeter.robotics import robot_registry

    csv_path = Path(csv_path).expanduser().resolve(strict=True)
    target_path = Path(target_path).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new replay output path: {output}")
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Replay output must be an MP4")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Replay FPS must be positive")
    scene = _load_scene(hoi_path, object_mesh)
    table = _synthetic_table(scene, table_size)
    frame_count, joint_names = _csv_shape(csv_path)
    if len(scene["translations"]) != frame_count:
        raise ValueError(f"Robot/object frame mismatch: {frame_count}/{len(scene['translations'])}")
    manifest = json.loads((target_path / "manifest.json").read_text())
    if manifest.get("name") != "unitree_g1_dex3":
        raise ValueError("Replay target must be unitree_g1_dex3")

    root = Path(soma_root).expanduser().resolve(strict=True)
    robot_registry.register_robots_path(str(target_path))
    buffer = csv_io.load_csv(str(csv_path), fps=float(fps))
    robot_builder = pipeline_utils.create_robot_builder("unitree_g1_dex3")
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_world(robot_builder, wp.transform_identity())
    model = builder.finalize(device=device)
    if model.joint_coord_count != 50 or model.articulation_count != 1:
        raise RuntimeError("Replay model is not the native 50-coordinate G1+DEx3 model")
    state = model.state()

    ffmpeg_log = output.with_suffix(".ffmpeg.log")
    ffmpeg_stream = ffmpeg_log.open("wb")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=ffmpeg_stream,
        stderr=subprocess.STDOUT,
    )
    viewer = None
    try:
        with wp.ScopedDevice(device):
            viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=True)
            viewer.set_model(model)
            viewer.set_world_offsets([0.0, 0.0, 0.0])
            viewer.camera.pos = viewer.camera._as_vec3(camera)
            viewer.camera.look_at(look_at)
            viewer.camera.fov = 42.0
            viewer.renderer.set_title("GRAIL SOMA G1+DEx3 scene replay")
            object_indices = wp.array(scene["faces"].reshape(-1), dtype=wp.int32, device=device)
            table_points = wp.array(table["vertices"], dtype=wp.vec3, device=device)
            table_indices = wp.array(table["faces"].reshape(-1), dtype=wp.int32, device=device)
            for frame in range(frame_count):
                coordinates = buffer.get_data(frame)
                wp.copy(model.joint_q, wp.array(coordinates, dtype=wp.float32, device=device))
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                object_vertices = (scene["vertices"] * scene["scale"]) @ scene["rotations"][
                    frame
                ].T + scene["translations"][frame]
                object_points = wp.array(object_vertices, dtype=wp.vec3, device=device)
                viewer.begin_frame(frame / fps)
                viewer.log_state(state)
                viewer.log_mesh(
                    "manipulated_object",
                    object_points,
                    object_indices,
                    color=(0.95, 0.42, 0.08),
                    roughness=0.65,
                    backface_culling=False,
                )
                viewer.log_mesh(
                    "support_table",
                    table_points,
                    table_indices,
                    color=(0.34, 0.22, 0.12),
                    roughness=0.8,
                    backface_culling=False,
                )
                viewer.end_frame()
                pixels = viewer.get_frame(render_ui=False).numpy()
                assert encoder.stdin is not None
                encoder.stdin.write(pixels.tobytes())
        assert encoder.stdin is not None
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait(timeout=120)
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}; see {ffmpeg_log}")
    except BaseException:
        if encoder.stdin is not None:
            encoder.stdin.close()
            encoder.stdin = None
        encoder.wait(timeout=120)
        raise
    finally:
        if viewer is not None:
            viewer.close()
        ffmpeg_stream.close()

    receipt = {
        "schema": "grail.soma_g1_dex3_scene_replay.v1",
        "status": "KINEMATIC_SCENE_REPLAY_UNREVIEWED",
        "csv": _binding(csv_path),
        "hoi": _binding(scene["source"]),
        "object_mesh": _binding(scene["mesh_path"]),
        "target_manifest": _binding(target_path / "manifest.json"),
        "video": _binding(output),
        "frames": frame_count,
        "fps": fps,
        "resolution": [width, height],
        "joint_count": len(joint_names),
        "object_scale": scene["scale"].tolist(),
        "table_source": "synthetic_under_initial_object",
        "table_pos": table["pos"].tolist(),
        "table_size": table["size"].tolist(),
        "table_surface_z": table["surface_z"],
        "table_leg_count": table["leg_count"],
        "camera": {"position": list(camera), "look_at": list(look_at)},
        "limitations": [
            "Kinematic visualization only; no physics or collision response.",
            "Object material textures are not loaded; source geometry and trajectory are loaded.",
            "Visual inspection is required before accepting contact or grasp quality.",
        ],
    }
    receipt_path = output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _triplet(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three values")
    return tuple(float(value) for value in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="50-column G1+DEx3 CSV")
    parser.add_argument("hoi", type=Path, help="Validated GRAIL hoi_data.pkl")
    parser.add_argument("target", type=Path, help="Source-bound G1+DEx3 target directory")
    parser.add_argument("output", type=Path, help="New MP4 path")
    parser.add_argument("--soma-root", type=Path, default=Path("imports/soma-retargeter"))
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument(
        "--table-size",
        nargs=3,
        default=(2.0, 0.6, 0.04),
        metavar=("WIDTH", "DEPTH", "THICKNESS"),
        help="Synthetic tabletop dimensions in meters (default: 2.0 0.6 0.04)",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", nargs=3, default=(2.35, -2.65, 1.75))
    parser.add_argument("--look-at", nargs=3, default=(0.1, 0.15, 0.78))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.camera = _triplet(args.camera)
    args.look_at = _triplet(args.look_at)
    args.table_size = _triplet(args.table_size)
    options = vars(args)
    options["csv_path"] = options.pop("csv")
    options["hoi_path"] = options.pop("hoi")
    options["target_path"] = options.pop("target")
    print(json.dumps(render_scene(**options), indent=2))


if __name__ == "__main__":
    main()
