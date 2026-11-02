#!/usr/bin/env python3
"""Overlay saved Sapiens observations from a native GEP/GEM + SOMA motion.

This renders image detections, not projected SOMA joints. No inference runs.
The original video timeline and a single fixed upper-body crop are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Published sapiens70_nofinger30_v1 -> COCO17, with wrists retained.
COCO17 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 21, 9, 10, 11, 12, 13, 14)
BODY_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
GREEN, YELLOW, CYAN = (85, 235, 110), (30, 220, 255), (255, 230, 30)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_validity(data, threshold, conditions_path):
    """Recompute a boolean mask on the source timeline; never interpolate masks."""
    import torch

    conditions = torch.load(conditions_path, map_location="cpu", weights_only=False)
    native_points = np.asarray(conditions["kp2d"])
    boxes = np.asarray(conditions["bbx_xys"])
    if not np.array_equal(native_points, data["native_observations"]["keypoints_2d"]):
        raise ValueError("Native crop conditions do not match the saved observations")
    if (
        boxes.shape != (len(native_points), 3)
        or not np.isfinite(boxes).all()
        or (boxes[:, 2] <= 0).any()
    ):
        raise ValueError("Invalid native person crop center/size")
    provenance = data["estimator_provenance"]
    native_times = np.asarray(provenance["native_timestamps_seconds"], dtype=np.float64)
    source_times = np.asarray(provenance["source_timestamps_seconds"], dtype=np.float64)
    points = np.asarray(data["sapiens30_keypoints_2d"])
    for times, count in ((native_times, len(boxes)), (source_times, len(points))):
        if (
            times.shape != (count,)
            or not np.isfinite(times).all()
            or not (np.diff(times) > 0).all()
        ):
            raise ValueError("Invalid recorded observation timestamps")
    native_times -= native_times[0]
    source_times -= source_times[0]
    source_boxes = np.stack(
        [np.interp(source_times, native_times, boxes[:, axis]) for axis in range(3)], axis=-1
    ).astype(np.float32)

    def mask(keypoints, crops):
        low = crops[:, None, :2] - crops[:, None, 2:] / 2
        high = crops[:, None, :2] + crops[:, None, 2:] / 2
        inside = ((keypoints[..., :2] >= low) & (keypoints[..., :2] <= high)).all(-1)
        return np.isfinite(keypoints).all(-1) & (keypoints[..., 2] > threshold) & inside

    return mask(points, source_boxes), source_boxes, mask(native_points, boxes)


def draw_points(panel, points, scale, offset, *, detail=False, valid=None):
    if valid is None:
        valid = points[:, 2] > 0
    xy = points[:, :2] * scale + np.asarray(offset)
    pixels = np.rint(xy).astype(int)
    radius = 5 if detail else 3
    for left, right in BODY_EDGES:
        a, b = COCO17[left], COCO17[right]
        if not (valid[a] and valid[b]):
            continue
        color = GREEN if min(points[a, 2], points[b, 2]) >= 0.5 else YELLOW
        cv2.line(panel, tuple(pixels[a]), tuple(pixels[b]), color, 2, cv2.LINE_AA)
    for index, point in enumerate(pixels):
        if not valid[index]:
            continue
        color = GREEN if points[index, 2] >= 0.5 else YELLOW
        if index in (21, 22):
            color = CYAN
        cv2.circle(panel, tuple(point), radius + 1, (15, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(point), radius, color, -1, cv2.LINE_AA)
        if detail and index in (21, 22):
            name = "R wrist" if index == 21 else "L wrist"
            where = tuple(point + [10, -12 if index == 21 else 22])
            cv2.putText(
                panel, name, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(panel, name, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validity-threshold",
        type=float,
        help="Show only score > threshold AND inside the person crop",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        help="Matching native conditions.pt; defaults to saved provenance path",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh visualization output directory")
    with np.load(args.motion, allow_pickle=True) as saved:
        data = saved["motion_incam"].item()
    points = np.asarray(data["sapiens30_keypoints_2d"])
    provenance = data["estimator_provenance"]
    if provenance["observation_source"] != "sapiens70_nofinger30_v1":
        raise ValueError("Unsupported keypoint observation schema")
    if provenance["source_sha256"] != sha256(args.video):
        raise ValueError("Source video differs from the bound detections")
    if points.ndim != 3 or points.shape[1:] != (30, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite frame x 30 x (x,y,score) observations")
    if not np.array_equal(points[:, COCO17], data["vitpose"]):
        raise ValueError("Saved COCO17 mapping differs from the observation schema")
    fps = float(data["mocap_frame_rate"])
    valid = points[..., 2] > 0
    validity_report = {"applied": False}
    conditions_path, source_boxes, native_valid = None, None, None
    if args.validity_threshold is not None:
        if not np.isfinite(args.validity_threshold) or args.validity_threshold < 0:
            raise ValueError("Validity threshold must be finite and nonnegative")
        conditions_path = args.conditions or Path(provenance["native_output_dir"]) / "conditions.pt"
        valid, source_boxes, native_valid = source_validity(
            data, args.validity_threshold, conditions_path
        )
        validity_report = dict(
            applied=True,
            score_threshold=args.validity_threshold,
            comparison="strict greater than",
            person_crop_required=True,
            crop_boundary="inclusive",
            source_alignment="Interpolate native crop center/size to recorded source times; recompute mask using saved source detections; binary masks are not interpolated",
            source_accepted=int(valid.sum()),
            source_total=int(valid.size),
            source_accepted_per_frame=valid.sum(-1).tolist(),
            right_wrist_accepted=int(valid[:, 21].sum()),
            left_wrist_accepted=int(valid[:, 22].sum()),
            native_accepted=int(native_valid.sum()),
            native_total=int(native_valid.size),
            inference_threshold_changed=False,
        )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - fps) > 1e-4:
        raise ValueError("Source video and saved motion timing differ")
    sw, sh = [
        int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
    ]
    width = 960
    height = 2 * round(sh * width / sw / 2)
    zoom_width = 480
    # Fixed image-space crop from all source frames; no camera or mesh edits.
    upper = points[:, COCO17[:13], :2].reshape(-1, 2)
    low, high = upper.min(0) - 30, upper.max(0) + 30
    center = (low + high) / 2
    crop_h = max(high[1] - low[1], (high[0] - low[0]) * height / zoom_width)
    crop_w = crop_h * zoom_width / height
    low = center - [crop_w / 2, crop_h / 2]
    zoom_scale = height / crop_h
    transform = np.array(
        [[zoom_scale, 0, -low[0] * zoom_scale], [0, zoom_scale, -low[1] * zoom_scale]]
    )
    args.output.mkdir(parents=True)
    if source_boxes is not None:
        np.savez_compressed(
            args.output / "validity_mask.npz",
            source_valid=valid,
            source_person_crops=source_boxes,
            native_valid=native_valid,
        )
    raw = args.output / "keypoints_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + zoom_width, height + 100)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")
    review_frames = sorted(
        {0, len(points) // 4, len(points) // 2, 3 * len(points) // 4, len(points) - 1}
    )
    snapshots = []
    for frame, keypoints in enumerate(points):
        ok, original = capture.read()
        if not ok:
            raise ValueError("Source video is shorter than the detections")
        overview = draw_points(
            cv2.resize(original, (width, height)), keypoints, width / sw, [0, 0], valid=valid[frame]
        )
        zoom = draw_points(
            cv2.warpAffine(original, transform, (zoom_width, height)),
            keypoints,
            zoom_scale,
            -low * zoom_scale,
            detail=True,
            valid=valid[frame],
        )
        canvas = np.full((height + 100, width + zoom_width, 3), (25, 29, 32), np.uint8)
        canvas[60 : 60 + height] = np.hstack((overview, zoom))
        cv2.putText(
            canvas,
            "2D detections | GEP/GEM + SOMA pipeline",
            (18, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Sapiens: {int(valid[frame].sum())}/30 shown | Frame {frame:03d} / {len(points) - 1:03d} | {fps:g} fps",
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (195, 205, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Upper-body detail (fixed crop)",
            (width + 12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                "Green: landmarks / body links   Cyan: wrists   Yellow: score < 0.5   |   No finger detections"
                if args.validity_threshold is None
                else f"Mask: score > {args.validity_threshold:.2f} + inside person crop   |   Green: landmarks   Cyan: wrists   |   No finger detections"
            ),
            (18, height + 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (width, 60), (width, height + 60), (25, 29, 32), 3)
        writer.write(canvas)
        if frame in review_frames:
            cv2.imwrite(str(args.output / f"frame_{frame:03d}.jpg"), canvas)
            snapshots.append(cv2.resize(canvas, (720, (height + 100) // 2)))
    if capture.read()[0]:
        raise ValueError("Source video is longer than the detections")
    capture.release()
    writer.release()
    output = args.output / "soma_pipeline_2d_keypoints.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    cv2.imwrite(str(args.output / "keypoint_contact_sheet.jpg"), np.vstack(snapshots))
    report = dict(
        scope="Saved 2D image detections; no projected SOMA joints or new inference",
        detector="Sapiens2",
        schema=provenance["observation_source"],
        frames=len(points),
        fps=fps,
        original_native_frames=len(data["native_observations"]["keypoints_2d"]),
        original_native_fps=data["native_observations"]["fps"],
        observation_resampling="existing source-aligned observations reused unchanged",
        score_range=[float(points[..., 2].min()), float(points[..., 2].max())],
        scores_are_not_calibrated_probabilities=True,
        wrists={"right": 21, "left": 22},
        finger_detections=False,
        crop_xywh=[*low.tolist(), float(crop_w), float(crop_h)],
        review_frames=review_frames,
        validity_mask=validity_report,
        source_inputs={
            str(p.resolve()): sha256(p) for p in [args.motion, args.video, Path(__file__)]
        },
        output_sha256=sha256(output),
    )
    if conditions_path is not None:
        report["source_inputs"][str(conditions_path.resolve())] = sha256(conditions_path)
        report["validity_mask_sha256"] = sha256(args.output / "validity_mask.npz")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(points), "fps": fps}))


if __name__ == "__main__":
    main()
