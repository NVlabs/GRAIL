"""Offline MoSE3 pixel tracks converted to metric poses of the original CAD.

Run in the MoSE3 environment; the pipeline may invoke this module as a subprocess.
Only the supplied first-frame pose, intrinsics, mesh and masks determine anchors.
No later FoundationPose poses, masks, observed depth, or trajectory smoothing are
used. Model inference is dense regardless of the number of sampled CAD anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from grail.adapters.mose3_pnp import estimate_pose_proposals
from grail.adapters.mose3_sampling import select_query_indices


def _binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    rotation = pose[:3, :3]
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Initial pose must be a finite proper SE3")
    return pose


def _map_pixels(pixels, affine):
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous = np.concatenate([pixels, np.ones((*pixels.shape[:-1], 1))], axis=-1)
    mapped = homogeneous @ affine.T
    return mapped[..., :2] / mapped[..., 2:]


def _raster_depth(vertices_camera, faces, K, hw):
    """Original triangle front surface at integer pixel centers, camera-Z meters."""
    if not np.isfinite(vertices_camera).all() or np.any(vertices_camera[:, 2] <= 1e-5):
        raise ValueError("Initial CAD is nonfinite or crosses the camera near plane")
    h, w = hw
    homogeneous = vertices_camera @ K.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    depth = np.full((h, w), np.inf)
    for face in faces:
        z, xy = vertices_camera[face, 2], uv[face]
        lo = np.maximum(np.ceil(xy.min(0)).astype(int), [0, 0])
        hi = np.minimum(np.floor(xy.max(0)).astype(int), [w - 1, h - 1])
        if np.any(lo > hi):
            continue
        den = (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (
            xy[0, 1] - xy[2, 1]
        )
        if abs(den) < 1e-12:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        a = (
            (xy[1, 1] - xy[2, 1]) * (xx - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (yy - xy[2, 1])
        ) / den
        b = (
            (xy[2, 1] - xy[0, 1]) * (xx - xy[2, 0]) + (xy[0, 0] - xy[2, 0]) * (yy - xy[2, 1])
        ) / den
        c = 1 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        zz = 1 / np.maximum(a / z[0] + b / z[1] + c / z[2], 1e-12)
        tile = depth[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        np.minimum(tile, np.where(inside, zz, np.inf), out=tile)
    return depth


def prepare_query_anchors(
    vertices_scaled_m,
    faces,
    initial_pose,
    K,
    native_hw,
    network_hw,
    object_mask0,
    human_mask0,
    max_queries=2048,
    sampling="raster_uniform",
    erosion_radius=1,
):
    """Deterministic spatial sampling of first-frame visible original CAD only.

    ``raster_uniform`` exactly retains the reviewed 2048-query sampling policy;
    ``farthest`` is an optional spatial-diversity alternative. Visibility logits
    are applied later by PnP, independently for frame zero and the current frame.
    Frontmost CAD rasterization is geometric visibility, not an occlusion label.
    """
    T0 = _validate_pose(initial_pose)
    vertices, faces, K = map(np.asarray, (vertices_scaled_m, faces, K))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_scaled_m must have shape (V,3)")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype.kind not in "iu"
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("faces must index original triangular geometry")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid intrinsics")
    sh, sw = map(int, native_hw)
    h, w = map(int, network_hw)
    if min(sh, sw, h, w) <= 0 or erosion_radius < 0:
        raise ValueError("Invalid image dimensions or erosion radius")
    affine = np.array([[w / sw, 0, 0.5 * w / sw - 0.5], [0, h / sh, 0.5 * h / sh - 0.5], [0, 0, 1]])
    Knet = affine @ K
    camera = vertices @ T0[:3, :3].T + T0[:3, 3]
    depth = _raster_depth(camera, faces, Knet, (h, w))
    obj, human = np.asarray(object_mask0).squeeze(), np.asarray(human_mask0).squeeze()
    if obj.shape != (sh, sw) or human.shape != (sh, sw):
        raise ValueError("First-frame masks must match the native video dimensions")
    mask = (obj > 0) & ~(human > 0)
    yy, xx = np.mgrid[:h, :w]
    source = np.rint(_map_pixels(np.stack([xx, yy], -1), np.linalg.inv(affine))).astype(int)
    source[..., 0] = np.clip(source[..., 0], 0, sw - 1)
    source[..., 1] = np.clip(source[..., 1], 0, sh - 1)
    small = mask[source[..., 1], source[..., 0]]
    if erosion_radius:
        size = 2 * int(erosion_radius) + 1
        small = cv2.erode(small.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    yy, xx = np.where(small & np.isfinite(depth) & (depth > 0))
    pool = np.c_[xx, yy]
    selected = select_query_indices(pool, np.ones(len(pool)), max_queries, method=sampling)
    xy = pool[selected]
    rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(Knet).T
    camera_points = rays * (depth[xy[:, 1], xy[:, 0]] / rays[:, 2])[:, None]
    inverse = np.eye(4)
    inverse[:3, :3] = T0[:3, :3].T
    inverse[:3, 3] = -np.einsum("ij,j->i", inverse[:3, :3], T0[:3, 3])
    object_points = camera_points @ inverse[:3, :3].T + inverse[:3, 3]
    return {
        "pixels_xy": xy,
        "object_points": object_points,
        "camera_points": camera_points,
        "K_network": Knet,
        "native_to_network": affine,
        "T_camera_object_initial": T0,
        "K": K,
        "native_hw": np.asarray(native_hw),
        "network_hw": np.asarray(network_hw),
    }


def proposals_from_tracks(anchors, tracks_local, model_K, visibility_logits, **pnp_settings):
    """Reproduce the reviewed tracks_local/model_K -> native CAD-PnP bridge."""
    points = np.asarray(tracks_local, dtype=np.float64)
    logits = np.asarray(visibility_logits)
    if points.ndim != 3 or points.shape[-1] != 3 or logits.shape != points.shape[:2]:
        raise ValueError("Expected tracks_local [T,N,3] and visibility_logits [T,N]")
    projected = points @ np.asarray(model_K, dtype=np.float64).T
    positive = np.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    uv = np.full((*points.shape[:2], 2), np.nan)
    uv[positive] = projected[positive, :2] / projected[positive, 2:]
    uv = _map_pixels(uv, np.linalg.inv(anchors["native_to_network"]))
    return estimate_pose_proposals(
        anchors["object_points"],
        uv,
        anchors["K"],
        anchors["T_camera_object_initial"],
        np.where(positive, logits, -np.inf),
        anchors["native_hw"],
        **pnp_settings,
    )


def predict_proposals(
    input_path,
    video_path,
    output_dir,
    checkpoint,
    mose3_root,
    *,
    fps=None,
    max_edge=756,
    window_targets=120,
    max_queries=2048,
    sampling="raster_uniform",
    erosion_radius=1,
    pnp_settings=None,
):
    """Save fresh source-rate proposals and reproducible native track caches.

    The NPZ input contains vertices_scaled_m, faces, T_camera_object_initial, K,
    object_mask0 and human_mask0. All geometry is in original CAD coordinates and
    meters; this adapter applies no additional centering or scale. Each offline
    context has frame zero plus up to window_targets consecutive source frames.
    Existing output directories are rejected so stale tracks cannot be reused.
    """
    import torch

    started = time.monotonic()
    source_root = Path(mose3_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    input_path, video_path = Path(input_path).resolve(), Path(video_path).resolve()
    if not (source_root / "mose3/models/mose3.py").is_file():
        raise FileNotFoundError(f"Missing installed MoSE3 source: {source_root}")
    model_files = [checkpoint / "config.json", checkpoint / "model.safetensors"]
    for path in model_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(window_targets, max_edge) < 1:
        raise ValueError("window_targets and max_edge must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MoSE3 inference requires the configured CUDA runtime")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_hw = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if total < 1 or not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("Video has invalid frame count or FPS")
    if fps is not None and (not np.isfinite(fps) or not np.isclose(fps, video_fps, atol=0.01)):
        raise ValueError("Declared FPS disagrees with the source video")
    fps = video_fps if fps is None else float(fps)
    with np.load(input_path, allow_pickle=False) as packed:
        inputs = {
            key: packed[key]
            for key in (
                "vertices_scaled_m",
                "faces",
                "T_camera_object_initial",
                "K",
                "object_mask0",
                "human_mask0",
            )
        }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(
        max_edge=max_edge,
        window_targets=window_targets,
        max_queries=max_queries,
        sampling=sampling,
        erosion_radius=erosion_radius,
        pnp=pnp_settings or {},
    )
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("mose3_pnp.py"),
        Path(__file__).with_name("mose3_sampling.py"),
    ]
    native_sources = sorted((source_root / "mose3").rglob("*.py"))
    record = {
        "status": "RUNNING",
        "method": "offline_mose3_metric_cad_pnp",
        "settings": settings,
        "inputs": [_binding(input_path), _binding(video_path)],
        "checkpoint": [_binding(path) for path in model_files],
        "code": [_binding(path) for path in code_paths + native_sources],
        "runtime": {
            "python": sys.executable,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "frames": total,
        "fps": fps,
        "native_hw": list(native_hw),
        "windows": [],
        "online": False,
        "smoothing": False,
        "future_foundationpose_inputs": False,
        "visibility": "frame-zero CAD front surface and object mask minus human, then per-frame model logit > 0",
    }
    _write_json(out / "record.json", record)
    try:
        sys.path.insert(0, str(source_root))
        from mose3.models.mose3 import MoSE3
        from mose3.utils.video_io import load_video_clip

        torch.manual_seed(0)
        np.random.seed(0)
        cv2.setRNGSeed(0)
        torch.set_num_threads(4)
        model = MoSE3.from_pretrained(str(checkpoint), strict=True).cuda().eval()
        poses = np.full((total, 4, 4), np.nan)
        available, quality = np.zeros(total, bool), np.zeros(total, bool)
        per_frame, anchors = [None] * total, None
        starts = range(1, total, window_targets) if total > 1 else [1]
        for index, start in enumerate(starts):
            ids = [0] + list(range(start, min(total, start + window_targets)))
            video, loaded_ids, orig_wh = load_video_clip(
                video_path,
                num_frames=len(ids),
                start_frame=0,
                frame_interval=1,
                max_w=max_edge,
                max_h=max_edge,
                central_crop=False,
                frame_indices=ids,
            )
            if loaded_ids != ids or tuple(orig_wh[::-1]) != native_hw:
                raise ValueError("MoSE3 video decoder changed source frame IDs or dimensions")
            _, h, w, _ = video.shape
            if anchors is None:
                anchors = prepare_query_anchors(
                    inputs["vertices_scaled_m"],
                    inputs["faces"],
                    inputs["T_camera_object_initial"],
                    inputs["K"],
                    native_hw,
                    (h, w),
                    inputs["object_mask0"],
                    inputs["human_mask0"],
                    max_queries=max_queries,
                    sampling=sampling,
                    erosion_radius=erosion_radius,
                )
                np.savez_compressed(out / "anchors.npz", **anchors)
            if not np.array_equal(anchors["network_hw"], [h, w]):
                raise ValueError("Network image dimensions changed between windows")
            x, y = anchors["pixels_xy"].T
            tensor = torch.from_numpy(video).to("cuda").permute(0, 3, 1, 2).float()[None] / 255
            torch.cuda.synchronize()
            window_started = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(tensor, query_frame_idx=0)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - window_started
            tracks = predicted["tracks_local"][0, :, y, x].float().cpu().numpy()
            logits = predicted["visibility_logits"][0, :, y, x].float().cpu().numpy()
            fx, fy = predicted["track_focal"][0].float().cpu().tolist()
            model_K = np.array([[fx, 0, 0.5 * w - 0.5], [0, fy, 0.5 * h - 0.5], [0, 0, 1]])
            del predicted, tensor
            np.savez_compressed(
                out / f"window_{index:03d}_tracks.npz",
                tracks_local=tracks,
                visibility_logits=logits,
                model_K=model_K,
                frame_ids=np.asarray(ids),
            )
            result = proposals_from_tracks(anchors, tracks, model_K, logits, **(pnp_settings or {}))
            use = slice(None) if index == 0 else slice(1, None)
            dest = np.asarray(ids)[use]
            poses[dest] = result["T_camera_object"][use]
            available[dest], quality[dest] = (
                result["pose_available"][use],
                result["quality_pass"][use],
            )
            for dst, row in zip(dest, result["per_frame"][use]):
                per_frame[int(dst)] = dict(row, frame=int(dst), window=index)
            np.savez_compressed(
                out / "poses.npz",
                T_camera_object=poses,
                pose_available=available,
                quality_pass=quality,
                valid=quality,
                frame_ids=np.arange(total),
                fps=fps,
            )
            record["windows"].append(
                {
                    "index": index,
                    "frame_ids": ids,
                    "network_hw": [h, w],
                    "inference_seconds": elapsed,
                }
            )
            _write_json(out / "record.json", record)
        record.update(
            status="COMPLETE",
            per_frame=per_frame,
            pose_available_count=int(available.sum()),
            quality_pass_count=int(quality.sum()),
            wall_seconds=time.monotonic() - started,
            outputs=[_binding(path) for path in sorted(out.glob("*.npz"))],
        )
        _write_json(out / "record.json", record)
        return out / "poses.npz"
    except Exception as exc:
        record.update(
            status="FAILED",
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.monotonic() - started,
        )
        _write_json(out / "record.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mose3-root", required=True, type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-edge", type=int, default=756)
    parser.add_argument("--window-targets", type=int, default=120)
    parser.add_argument("--max-queries", type=int, default=2048)
    parser.add_argument(
        "--sampling", choices=("raster_uniform", "farthest"), default="raster_uniform"
    )
    parser.add_argument("--erosion-radius", type=int, default=1)
    parser.add_argument("--pnp-config", type=Path)
    args = parser.parse_args()
    pnp = None
    if args.pnp_config:
        import yaml

        pnp = yaml.safe_load(args.pnp_config.read_text())["pnp"]
    result = predict_proposals(
        args.input,
        args.video,
        args.output_dir,
        args.checkpoint,
        args.mose3_root,
        fps=args.fps,
        max_edge=args.max_edge,
        window_targets=args.window_targets,
        max_queries=args.max_queries,
        sampling=args.sampling,
        erosion_radius=args.erosion_radius,
        pnp_settings=pnp,
    )
    print(f"  MoSE3 metric proposals: {result}")


if __name__ == "__main__":
    main()
