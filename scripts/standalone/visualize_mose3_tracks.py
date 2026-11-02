"""Render cached raw MoSE3 point tracks over their source video, on CPU."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def binding(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def project_tracks(local, model_K, native_to_network):
    local = np.asarray(local, dtype=np.float64)
    projected = local @ np.asarray(model_K, dtype=np.float64).T
    positive = np.isfinite(local).all(axis=-1) & (local[..., 2] > 1e-6)
    uv = np.full((*local.shape[:-1], 2), np.nan)
    uv[positive] = projected[positive, :2] / projected[positive, 2:]
    homogeneous = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), axis=-1)
    native = homogeneous @ np.linalg.inv(native_to_network).T
    return native[..., :2] / native[..., 2:], positive


def text(image, line, x, y, scale=0.48, color=(235, 235, 235)):
    cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Encoder:
    def __init__(self, path, shape, fps):
        self.path = path
        h, w = shape
        if h % 2 or w % 2:
            raise ValueError("H264 dimensions must be even")
        self.process = subprocess.Popen(
            [
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, image):
        self.process.stdin.write(image.tobytes())

    def finish(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode()
        if self.process.wait() != 0:
            raise RuntimeError(error)

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--subset-indices", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="axe02")
    p.add_argument("--zoom-xyxy", nargs=4, type=int, default=[100, 350, 620, 1000])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier visualizations; use a new output directory")
    a_path = args.cache_dir / "anchors.npz"
    t_path = args.cache_dir / "window_000_tracks.npz"
    a, t = np.load(a_path, allow_pickle=False), np.load(t_path, allow_pickle=False)
    subset = np.load(args.subset_indices, allow_pickle=False)
    points, positive = project_tracks(t["tracks_local"], t["model_K"], a["native_to_network"])
    n, count = points.shape[:2]
    if not np.array_equal(t["frame_ids"], np.arange(n)):
        raise ValueError("Expected a complete original source timeline")
    if (
        subset.ndim != 1
        or len(np.unique(subset)) != len(subset)
        or subset.min() < 0
        or subset.max() >= count
    ):
        raise ValueError("Invalid query subset")
    visible = t["visibility_logits"] > 0
    initial = a["pixels_xy"].astype(float)
    # Spatial color coding stays attached to query identity through the sequence.
    scalar = initial[:, 0] + 0.35 * initial[:, 1]
    hue = np.rint(160 * (scalar - scalar.min()) / max(np.ptp(scalar), 1)).astype(np.uint8)
    hsv = np.stack([hue, np.full(count, 215, np.uint8), np.full(count, 255, np.uint8)], axis=-1)
    colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError("Invalid source FPS")
    x0, y0, x1, y1 = args.zoom_xyxy
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Zoom crop is outside the original frame")
    finite = np.isfinite(points).all(axis=-1) & positive
    inside = (
        finite
        & (points[..., 0] >= 0)
        & (points[..., 0] < w)
        & (points[..., 1] >= 0)
        & (points[..., 1] < h)
    )
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "projected_tracks.npz",
        pixels_native=points,
        positive_depth=positive,
        predicted_visible=visible,
        subset_indices=subset,
        colors_bgr=colors,
        frame_ids=t["frame_ids"],
    )
    bindings = [binding(x) for x in [__file__, a_path, t_path, args.video, args.subset_indices]]
    encoders, sheets, per_frame = {}, {"full": [], "zoom": []}, []
    keys = sorted(set([0, 30, 58, 60, 90, n - 1]))
    methods = [
        ("Original source RGB", None),
        (f"All {count} raw point tracks", np.arange(count)),
        (f"Diverse {len(subset)} tracks + short trails", subset),
    ]

    def draw_panel(frame, f, ids, crop, width):
        cx0, cy0, cx1, cy1 = crop
        scale = width / (cx1 - cx0)
        height = round((cy1 - cy0) * scale)
        height += height % 2
        base = cv2.resize(frame[cy0:cy1, cx0:cx1], (width, height), interpolation=cv2.INTER_AREA)
        sy = height / (cy1 - cy0)
        xy = (points - np.array([cx0, cy0])) * np.array([scale, sy])
        if ids is None:
            return base, "Original video; no pose rendering"
        crop_inside = (
            inside[f, ids]
            & (points[f, ids, 0] >= cx0)
            & (points[f, ids, 0] < cx1)
            & (points[f, ids, 1] >= cy0)
            & (points[f, ids, 1] < cy1)
        )
        small = len(ids) <= 256
        for i in ids:
            color = tuple(int(x) for x in colors[i])
            if small:
                for j in range(max(1, f - 5), f + 1):
                    if not (
                        inside[j - 1, i] and inside[j, i] and visible[j - 1, i] and visible[j, i]
                    ):
                        continue
                    p0, p1 = np.rint(xy[j - 1, i]).astype(int), np.rint(xy[j, i]).astype(int)
                    ok, q0, q1 = cv2.clipLine((0, 0, width, height), tuple(p0), tuple(p1))
                    if ok:
                        faded = tuple(
                            int(c * (0.25 + 0.6 * (j - max(1, f - 5) + 1) / 6)) for c in color
                        )
                        cv2.line(base, q0, q1, faded, 1, cv2.LINE_AA)
            if not inside[f, i]:
                continue
            x, y = np.rint(xy[f, i]).astype(int)
            if not (0 <= x < width and 0 <= y < height):
                continue
            if visible[f, i]:
                if small:
                    cv2.circle(base, (x, y), 3, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(base, (x, y), 2 if small else 1, color, -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(base, (x, y), (155, 155, 155), cv2.MARKER_TILTED_CROSS, 5, 1)
        detail = f"Visible {int(visible[f, ids].sum())}/{len(ids)} | in view {int(crop_inside.sum())}/{len(ids)}"
        return base, detail

    try:
        for f in range(n):
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (h, w):
                raise ValueError(f"Source decode failure at frame {f}")
            per_frame.append(
                {
                    "frame": f,
                    "all_predicted_visible": int(visible[f].sum()),
                    "all_in_source_view": int(inside[f].sum()),
                    "subset_predicted_visible": int(visible[f, subset].sum()),
                    "subset_in_source_view": int(inside[f, subset].sum()),
                }
            )
            for view, crop, width in [("full", (0, 0, w, h), 432), ("zoom", (x0, y0, x1, y1), 520)]:
                panels = []
                for label, ids in methods:
                    image, detail = draw_panel(frame, f, ids, crop, width)
                    header = np.full((58, width, 3), 22, np.uint8)
                    text(header, label, 10, 22, 0.49)
                    text(header, detail, 10, 46, 0.43, (180, 195, 205))
                    panels.append(np.concatenate([header, image]))
                body = np.concatenate(panels, axis=1)
                header = np.full((44, body.shape[1], 3), 12, np.uint8)
                text(
                    header,
                    f"{args.label} | MoSE3 point tracking | Frame {f:03d}/{n-1} | {f/fps:.3f}s | {'Fixed zoom' if view == 'zoom' else 'Full source view'}",
                    12,
                    28,
                    0.62,
                )
                footer = np.full((86, body.shape[1], 3), 12, np.uint8)
                text(
                    footer,
                    "Colors stay fixed to query identities; nearby points may share colors. Short trails span up to 6 frame steps.",
                    12,
                    22,
                )
                text(
                    footer,
                    "Filled dots: predicted visible (logit > 0). Gray crosses: predicted hidden. Points are shown even if CAD-PnP rejects the pose.",
                    12,
                    47,
                )
                text(
                    footer,
                    "Raw network tracks in the source camera. No query-offset correction, fitting, smoothing or interpolation. Visibility is a prediction.",
                    12,
                    72,
                    0.45,
                )
                canvas = np.concatenate([header, body, footer])
                if view not in encoders:
                    encoders[view] = Encoder(
                        args.output / f"{args.label}_point_tracks_{view}.mp4", canvas.shape[:2], fps
                    )
                encoders[view].write(canvas)
                if f in keys:
                    sheets[view].append(canvas)
                    cv2.imwrite(str(args.output / f"{view}_frame{f:03d}.jpg"), canvas)
        if cap.read()[0]:
            raise ValueError("Source has more frames than cached tracks")
        for enc in encoders.values():
            enc.finish()
    except BaseException:
        for enc in encoders.values():
            enc.abort()
        raise
    finally:
        cap.release()
    for view, frames in sheets.items():
        cv2.imwrite(str(args.output / f"{args.label}_{view}_keyframes.jpg"), np.concatenate(frames))
    meta = {
        "frames": n,
        "fps": fps,
        "native_hw": [h, w],
        "query_count": count,
        "subset_count": len(subset),
        "fixed_zoom_xyxy": args.zoom_xyxy,
        "color_policy": "Fixed color by original query pixel; identity persistent across frames",
        "projection": "tracks_local @ model_K, perspective divide, inverse pixel-center-aware native_to_network",
        "visibility_gate": "predicted logit > 0; not a calibrated probability or GT",
        "no_pnp_gate": True,
        "query_offset_correction": False,
        "trail_max_segments": 6,
        "bindings": bindings,
        "per_frame": per_frame,
        "videos": {},
        "code": binding(__file__),
    }
    for view, enc in encoders.items():
        c = cv2.VideoCapture(str(enc.path))
        decoded = 0
        out_fps = c.get(cv2.CAP_PROP_FPS)
        while c.read()[0]:
            decoded += 1
        c.release()
        if decoded != n or abs(out_fps - fps) > 1e-6:
            raise ValueError("Output video frame/FPS mismatch")
        meta["videos"][view] = {**binding(enc.path), "decoded_frames": decoded, "fps": out_fps}
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps({"output": str(args.output), "frames": n, "fps": fps, "videos": meta["videos"]})
    )


if __name__ == "__main__":
    main()
