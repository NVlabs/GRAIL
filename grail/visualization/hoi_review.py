"""Fast CPU review of exported native HOI geometry; no model loading or fitting.

Example (complete source-rate video plus representative stills)::

    python -m grail.visualization.hoi_review --geometry stage2/stage2_geometry.npz \
        --source-video source.mp4 --output-dir review --label 'Drill | Stage 2'

For intermediate review use ``--keyframes-only --frames 0,35,60,120``. Frame
numbers are zero-based source indices; this never creates a shortened video at
the original FPS. Without --keyframes-only the video always retains every source
frame. The cache holds only the hash-bound CPU raster binary, never motion data.
Use ``--camera-reference baseline_geometry.npz`` for the same oblique camera in
before/after comparisons; source projection still uses each checkpoint's camera.

NPZ contract: human_vertices [T,V,3], object_vertices [T,N,3], respective static
integer faces [F,3], camera_R [3,3], camera_T [3], focal (scalar or fx,fy), and
image_size [height,width]. Camera R/T use GRAIL's PyTorch3D row-vector convention.
World coordinates are already in the checkpoint's units; there is no rescaling.
This command consumes exported SOMA geometry.

Requirements: numpy, opencv-python, g++, ffmpeg and ffprobe. Images use original
triangles and human/object depth occlusion; neither rendered colors nor this
visualization receipt establish contact, collision, or reconstruction acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        geometry = {key: data[key] for key in (
            "human_vertices", "human_faces", "object_vertices", "object_faces",
            "camera_R", "camera_T", "focal", "image_size",
        )}
        for key in ("joints", "body_2d", "hand_2d"):
            if key in data:
                geometry[key] = data[key]
    count = len(geometry["human_vertices"])
    if count < 1:
        raise ValueError("Geometry must contain at least one frame")
    for kind in ("human", "object"):
        vertices = geometry[f"{kind}_vertices"]
        faces = geometry[f"{kind}_faces"]
        if vertices.ndim != 3 or vertices.shape[0] != count or vertices.shape[2] != 3:
            raise ValueError(f"{kind}_vertices must have shape [T,V,3] on the same timeline")
        if not np.isfinite(vertices).all() or not vertices.shape[1]:
            raise ValueError(f"Invalid {kind} vertices")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
            raise ValueError(f"{kind}_faces must be static integer [F,3]")
        if not len(faces) or faces.min() < 0 or faces.max() >= vertices.shape[1]:
            raise ValueError(f"Invalid {kind} face indices")
        geometry[f"{kind}_faces"] = np.ascontiguousarray(faces, dtype=np.int64)
    rotation = np.asarray(geometry["camera_R"], dtype=float).reshape(3, 3)
    translation = np.asarray(geometry["camera_T"], dtype=float).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Camera contains nonfinite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera_R is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("camera_R must be a proper rotation")
    focal = np.asarray(geometry["focal"], dtype=float).reshape(-1)
    if len(focal) not in (1, 2) or not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("focal must be a positive scalar or fx,fy")
    size = np.asarray(geometry["image_size"]).reshape(-1)
    if len(size) != 2 or not np.isfinite(size).all() or (size < 1).any():
        raise ValueError("image_size must be height,width")
    if not np.array_equal(size, size.astype(int)):
        raise ValueError("image_size must contain integers")
    geometry.update(camera_R=rotation, camera_T=translation,
                    focal=np.repeat(focal, 2) if len(focal) == 1 else focal,
                    image_size=size.astype(int))
    return geometry


def source_camera(geometry):
    """Convert PyTorch3D +X left/+Y up to raster +X right/+Y down."""
    flip = np.diag([-1.0, -1.0, 1.0])
    return geometry["camera_R"].reshape(3, 3) @ flip, geometry["camera_T"].reshape(3) @ flip


def fixed_oblique_camera(geometry, yaw_degrees=55.0):
    """One world-Z-up camera framed on the whole sequence, with no framewise chase."""
    arrays = (geometry["human_vertices"], geometry["object_vertices"])
    low = np.minimum.reduce([v.min(axis=(0, 1)) for v in arrays])
    high = np.maximum.reduce([v.max(axis=(0, 1)) for v in arrays])
    target = (low + high) / 2
    rotation, translation = source_camera(geometry)
    eye = -translation @ rotation.T
    angle = np.radians(yaw_degrees)
    yaw = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    direction = yaw @ (eye - target)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, -1.0, 0.5])
    # Bounding sphere and focal=1.1*panel fit all frames with margin.
    eye = target + direction / np.linalg.norm(direction) * max(0.1, 1.6 * np.linalg.norm(high-low))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=1)
    return rotation, -eye @ rotation


def select_frames(text, count):
    if text is None:
        return sorted(set(np.linspace(0, count - 1, min(6, count), dtype=int).tolist()))
    try:
        frames = sorted(set(int(item.strip()) for item in text.split(",")))
    except ValueError as error:
        raise ValueError("--frames requires comma-separated integer source indices") from error
    if not frames or frames[0] < 0 or frames[-1] >= count:
        raise ValueError(f"--frames must be within 0..{count-1}")
    return frames


def build_raster(cache_dir):
    """Locked, atomic, source/compiler/architecture-bound reusable C++ binary."""
    started = time.monotonic()
    source = Path(__file__).with_name("hoi_raster.cpp")
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        raise RuntimeError("A C++ compiler is required (g++ or CXX)")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    signature = {"source_sha256": binding(source)["sha256"],
                 "compiler": compiler, "compiler_version": compiler_version,
                 "platform": platform.platform(), "machine": platform.machine(),
                 "flags": ["-O3", "-shared", "-fPIC", "-std=c++11"]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir).expanduser().resolve() / key
    folder.mkdir(parents=True, exist_ok=True)
    library, metadata = folder / "raster.so", folder / "build.json"
    cache_hit = False
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if library.exists() and metadata.exists():
            try:
                record = json.loads(metadata.read_text())
                cache_hit = (record["signature"] == signature and
                             record["binary_sha256"] == binding(library)["sha256"])
            except (ValueError, KeyError):
                cache_hit = False
        if not cache_hit:
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                built = Path(temporary) / "raster.so"
                subprocess.run([compiler, *signature["flags"], str(source), "-o", str(built)],
                               check=True, capture_output=True, text=True)
                record = {"signature": signature, "binary_sha256": binding(built)["sha256"]}
                receipt = Path(temporary) / "build.json"
                receipt.write_text(json.dumps(record, indent=2) + "\n")
                os.replace(built, library)
                os.replace(receipt, metadata)
    return library, {"cache_hit": cache_hit, "seconds": time.monotonic()-started,
                     "signature": signature, "binary": binding(library)}


class CpuRasterizer:
    """Reusable output buffers; returned arrays remain valid until the next call."""

    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.hoi_raster
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
        self.function.restype = None
        self.buffers = {}

    def raster(self, vertices, faces, focal, height, width, slot="mesh"):
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if not np.isfinite(vertices).all():
            raise ValueError("Cannot rasterize nonfinite vertices")
        key = (slot, height, width)
        if key not in self.buffers:
            self.buffers[key] = (np.empty((height, width), np.float32),
                                 np.empty((height, width), np.int32))
        depth, ids = self.buffers[key]
        fx, fy = np.broadcast_to(np.asarray(focal).reshape(-1), (2,))
        self.function(vertices.ctypes.data, faces.ctypes.data, len(faces), fx, fy,
                      height, width, 1e-4, depth.ctypes.data, ids.ctypes.data)
        return depth, ids


def render_frame(rasterizer, geometry, index, rotation, translation, focal, height, width,
                 background=None):
    image = np.full((height, width, 3), 38, np.uint8) if background is None else background.copy()
    if image.shape != (height, width, 3):
        raise ValueError("Background must match the raster dimensions")
    meshes = {}
    for kind in ("human", "object"):
        vertices = geometry[f"{kind}_vertices"][index] @ rotation + translation
        faces = geometry[f"{kind}_faces"]
        depth, ids = rasterizer.raster(vertices, faces, focal, height, width, slot=kind)
        meshes[kind] = (vertices, faces, depth, ids)
    human_depth, object_depth = meshes["human"][2], meshes["object"][2]
    masks = {"human": np.isfinite(human_depth) & (human_depth < object_depth),
             "object": np.isfinite(object_depth) & (object_depth <= human_depth)}
    for kind, color in (("human", [235, 145, 60]), ("object", [35, 155, 245])):
        vertices, faces, _, ids = meshes[kind]
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                           vertices[faces[:, 2]] - vertices[faces[:, 0]])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = .45 + .55 * np.abs(normals @ (np.array([.2, -.3, .93]) / np.linalg.norm([.2, -.3, .93])))
        colors = light[:, None] * np.asarray(color)
        mask = masks[kind]
        alpha = 1.0 if background is None else 0.65
        image[mask] = (alpha * colors[ids[mask]] + (1-alpha) * image[mask]).astype(np.uint8)
    return image


def make_review(geometry_path, source_video, output_dir, *, label="HOI reconstruction",
                frames=None, keyframes_only=False, cache_dir=None, panel_size=640,
                camera_reference=None):
    import cv2

    started = time.monotonic()
    if panel_size < 128 or panel_size % 2:
        raise ValueError("panel_size must be an even integer >= 128")
    if not label.isascii():
        raise ValueError("Use an English/ASCII label; OpenCV text does not support Unicode")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory to preserve earlier reviews")
    geometry = load_geometry(geometry_path)
    count = len(geometry["human_vertices"])
    selected = select_frames(frames, count)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video: {source_video}")
    source_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    height, width = map(int, geometry["image_size"])
    source_size = (round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if source_count != count or source_size != (height, width) or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Source/geometry timeline or size mismatch: {source_count}, {source_size}, {fps}")
    library, build = build_raster(cache_dir or Path.home()/".cache/grail/hoi_review")
    rasterizer = CpuRasterizer(library)
    source_rotation, source_translation = source_camera(geometry)
    camera_geometry = load_geometry(camera_reference) if camera_reference else geometry
    oblique_rotation, oblique_translation = fixed_oblique_camera(camera_geometry)
    scale = min(panel_size/height, panel_size/width)
    render_height, render_width = round(height*scale), round(width*scale)
    # Match actual resized pixel aspect even when rounding differs between axes.
    source_focal = geometry["focal"] * [render_width/width, render_height/height]
    margin, video_width, video_height = 96, 2*panel_size, panel_size+96
    times = {"load_and_setup_seconds": time.monotonic()-started,
             "decode_seconds": 0.0, "render_seconds": 0.0, "encode_write_seconds": 0.0}
    video = output / "source_and_oblique.mp4"
    process = None
    stills = []
    if not keyframes_only:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{video_width}x{video_height}", "-r", str(fps),
                                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
                                    "-crf", "19", "-threads", "2", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    render_indices = selected if keyframes_only else range(count)
    # Some generated MP4s report successful seeks but decode an earlier keyframe.
    # Decode sequentially so stills use the same source frame as native geometry.
    decode_indices = range(max(selected) + 1) if keyframes_only else render_indices
    selected_set = set(selected)
    try:
        for index in decode_indices:
            stamp = time.monotonic()
            ok, background = cap.read()
            if not ok:
                raise ValueError(f"Missing source frame {index}")
            if keyframes_only and index not in selected_set:
                times["decode_seconds"] += time.monotonic()-stamp
                continue
            background = cv2.resize(background, (render_width, render_height), interpolation=cv2.INTER_AREA)
            times["decode_seconds"] += time.monotonic()-stamp
            stamp = time.monotonic()
            source = render_frame(rasterizer, geometry, index, source_rotation, source_translation,
                                  source_focal, render_height, render_width, background)
            oblique = render_frame(rasterizer, geometry, index, oblique_rotation, oblique_translation,
                                   np.array([1.1*panel_size]*2), panel_size, panel_size)
            canvas = np.full((video_height, video_width, 3), (24, 28, 34), np.uint8)
            x, y = (panel_size-render_width)//2, margin+(panel_size-render_height)//2
            canvas[y:y+render_height, x:x+render_width] = source
            canvas[margin:, panel_size:] = oblique
            texts = [(f"{label} | UNREVIEWED", (12, 25)),
                     (f"Source overlay | Frame {index}/{count-1} | {index/fps:.2f}s", (12, 52)),
                     ("Blue: human | Orange: object | Native meshes", (12, 78)),
                     ("Fixed oblique view", (panel_size+12, 25)),
                     ("Visual evidence only; contact acceptance not assessed", (panel_size+12, 52))]
            text_scale = min(.55, panel_size/1100)
            for text, position in texts:
                cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                            (225, 232, 241), 1, cv2.LINE_AA)
            times["render_seconds"] += time.monotonic()-stamp
            stamp = time.monotonic()
            if process:
                process.stdin.write(canvas.tobytes())
            if index in selected:
                path = output / f"frame_{index:05d}.jpg"
                if not cv2.imwrite(str(path), canvas):
                    raise RuntimeError(f"Failed writing {path}")
                stills.append((index, path, cv2.resize(canvas, (640, round(video_height*640/video_width)))))
            times["encode_write_seconds"] += time.monotonic()-stamp
        if not keyframes_only and cap.read()[0]:
            raise ValueError("Source contains more decoded frames than the geometry")
    finally:
        cap.release()
        if process:
            process.stdin.close()
            exit_code = process.wait()
    if process and exit_code:
        raise RuntimeError(f"ffmpeg exited with {exit_code}")
    rows = []
    for start in range(0, len(stills), 2):
        images = [item[2] for item in stills[start:start+2]]
        if len(images) == 1:
            images.append(np.full_like(images[0], 24))
        rows.append(np.concatenate(images, axis=1))
    sheet = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(sheet), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"Failed writing {sheet}")
    stream = None
    if not keyframes_only:
        stream = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json",
            str(video)], text=True))["streams"][0]
        numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
        if int(stream["nb_read_frames"]) != count or abs(numerator/denominator-fps) > 1e-4:
            raise ValueError("Encoded review timeline differs from source")
    receipt = {"schema": "grail.hoi_cpu_review.v1", "status": "RENDER_COMPLETE_UNREVIEWED",
               "geometry": binding(geometry_path), "source_video": binding(source_video),
               "code": binding(__file__), "raster_source": binding(Path(__file__).with_name("hoi_raster.cpp")),
               "raster_build": build, "frame_count": count, "fps": fps,
               "rendered_frames": list(render_indices), "still_frames": selected,
               "mode": "keyframes_only" if keyframes_only else "full_video",
               "source_camera_convention": "GRAIL PyTorch3D row vectors; raster flips camera X and Y",
               "oblique_camera_reference": binding(camera_reference or geometry_path),
               "fixed_oblique_camera": {"R": oblique_rotation.tolist(), "T": oblique_translation.tolist(),
                                         "focal": 1.1*panel_size, "world_up": "+Z"},
               "outputs": {"sheet": binding(sheet), "stills": [binding(item[1]) for item in stills]},
               "timing": times, "notes": ["Original vertices/faces; no motion or mesh edits.",
                "CPU visual review only, not contact, collision, or quality acceptance.",
                "Source and fixed oblique views are rasterized at the declared display resolution."],
               "panel_size": panel_size}
    if not keyframes_only:
        receipt["outputs"]["video"] = binding(video)
        receipt["encoded_stream"] = stream
    receipt["timing"]["total_seconds"] = time.monotonic()-started
    (output/"review_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="HOI reconstruction")
    parser.add_argument("--frames", help="Comma-separated zero-based source indices for stills")
    parser.add_argument("--keyframes-only", action="store_true", help="Render stills only; omit full video")
    parser.add_argument("--cache-dir", type=Path, help="CPU compiler cache (default ~/.cache/grail/hoi_review)")
    parser.add_argument("--panel-size", type=int, default=640)
    parser.add_argument("--camera-reference", type=Path,
                        help="Baseline geometry NPZ defining a shared fixed oblique camera")
    args = parser.parse_args()
    receipt = make_review(args.geometry, args.source_video, args.output_dir, label=args.label,
                          frames=args.frames, keyframes_only=args.keyframes_only,
                          cache_dir=args.cache_dir, panel_size=args.panel_size,
                          camera_reference=args.camera_reference)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output_dir/"review_receipt.json"),
                      "seconds": receipt["timing"]["total_seconds"]}))


if __name__ == "__main__":
    main()
