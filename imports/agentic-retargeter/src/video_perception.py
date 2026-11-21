"""Local video evidence extraction; no remote inference or contact truth claims.

Grounding DINO locates a text-named object, SAM2 predicts its visible mask, and
MediaPipe estimates hand landmarks. Model directories must already exist locally.
The output is image-space evidence for the temporal planner, never 3D contacts.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_binding(path: str | Path) -> dict:
    """Bind local model assets without depending on a mutable model name."""
    path = Path(path).expanduser().resolve(strict=True)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file()
        and p.suffix in {".json", ".txt", ".safetensors", ".bin", ".task"}
    )
    if not files:
        raise ValueError(f"No model assets in {path}")
    return {"path": str(path), "files": [
        {"name": p.name if path.is_file() else str(p.relative_to(path)),
         "sha256": sha256_file(p)} for p in files
    ]}


def read_video_timing(path: str | Path, *, ffprobe: str = "ffprobe") -> dict:
    """Read actual presentation timestamps; do not invent constant-rate timing."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_streams", "-show_entries",
         "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
         "-of", "json", str(path)], check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected video stream")
    stream = streams[0]
    pts = [float(f["best_effort_timestamp_time"]) for f in data.get("frames", [])]
    if not pts or not all(math.isfinite(t) for t in pts):
        raise ValueError("Video has no finite presentation timestamps")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Video timestamps must be strictly increasing")
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video frame rate")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": fps, "frame_count": len(pts), "pts_origin_seconds": pts[0],
            "time_seconds": [t - pts[0] for t in pts], "timing_basis": "ffprobe_pts"}


def box_iou(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    extent = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    overlap = float(np.prod(extent))
    union = float(np.prod(np.maximum(0, a[2:] - a[:2]))
                  + np.prod(np.maximum(0, b[2:] - b[:2])) - overlap)
    return overlap / union if union > 0 else 0.0


def choose_object(boxes: Any, scores: Any, previous_box=None) -> tuple[int | None, bool]:
    """Select one object, flag near ties instead of silently asserting identity."""
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    if scores.size == 0:
        return None, False
    if boxes.shape != (len(scores), 4) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid detector output")
    ranking = scores.copy()
    if previous_box is not None:
        ranking += np.array([0.3 * box_iou(b, previous_box) for b in boxes])
    order = np.argsort(ranking)[::-1]
    ambiguous = bool(len(order) > 1 and ranking[order[0]] - ranking[order[1]] < .08
                     and box_iou(boxes[order[0]], boxes[order[1]]) < .5)
    return int(order[0]), ambiguous


def clip_detection_boxes(boxes: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip image boxes, dropping nonfinite/degenerate proposals explicitly."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    valid = np.isfinite(boxes).all(axis=1) & (clipped[:, 2:] > clipped[:, :2]).all(axis=1)
    return clipped, valid


class ObjectTrack:
    """Do not let an uncertain first selection become a confident identity prior."""

    def __init__(self):
        self.previous_box = None

    def select(self, boxes, scores):
        index, ambiguous = choose_object(boxes, scores, self.previous_box)
        self.previous_box = None if index is None or ambiguous else np.asarray(boxes[index]).tolist()
        return index, ambiguous


def object_context_crop(box, width: int, height: int) -> tuple[int, int, int, int]:
    """Automatic target context, increasing hand resolution without manual ROIs."""
    padding = max(.075 * max(width, height), .4 * max(box[2] - box[0], box[3] - box[1]))
    return (max(0, int(math.floor(box[0] - padding))),
            max(0, int(math.floor(box[1] - padding))),
            min(width, int(math.ceil(box[2] + padding))),
            min(height, int(math.ceil(box[3] + padding))))


class HandTracks:
    """Short-gap wrist association; detector order and handedness are not IDs."""

    def __init__(self, max_gap_seconds: float = .25):
        self.max_gap_seconds = max_gap_seconds
        self.previous: dict[str, dict] = {}
        self.next_id = 0

    def update(self, hands: list[dict], time_seconds: float, image_size: tuple[int, int]) -> list[dict]:
        self.previous = {k: v for k, v in self.previous.items()
                         if time_seconds - v["time"] <= self.max_gap_seconds}
        diagonal = float(np.hypot(*image_size))
        available = set(self.previous)
        pairs = []
        for i, hand in enumerate(hands):
            wrist = np.asarray(hand["landmarks_xy"][0])
            for track_id, record in self.previous.items():
                distance = float(np.linalg.norm(wrist - record["wrist"])) / diagonal
                if distance <= .10:
                    pairs.append((distance, i, track_id))
        assigned = {}
        for _, i, track_id in sorted(pairs):
            if i not in assigned and track_id in available:
                assigned[i] = track_id
                available.remove(track_id)
        result = []
        for i, hand in enumerate(hands):
            if i not in assigned:
                assigned[i] = f"hand_{self.next_id:03d}"
                self.next_id += 1
            track_id = assigned[i]
            item = {**hand, "track_id": track_id}
            self.previous[track_id] = {"time": time_seconds,
                                       "wrist": np.asarray(hand["landmarks_xy"][0])}
            result.append(item)
        return result


class LocalPerception:
    """Optional heavyweight dependencies are loaded only on explicit use."""

    def __init__(self, *, detector_dir, segmenter_dir, hand_model, device="cpu",
                 detection_threshold=.25, text_threshold=.20, hand_threshold=.30):
        import torch
        import mediapipe as mp
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor

        for path in (detector_dir, segmenter_dir, hand_model):
            if not Path(path).expanduser().exists():
                raise FileNotFoundError(f"Local model asset required: {path}")
        detector_dir, segmenter_dir, hand_model = [
            str(Path(p).expanduser().resolve()) for p in (detector_dir, segmenter_dir, hand_model)]
        for name, value in (("detection_threshold", detection_threshold),
                            ("text_threshold", text_threshold), ("hand_threshold", hand_threshold)):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        self.torch, self.mp, self.device = torch, mp, device
        self.detection_threshold, self.text_threshold = detection_threshold, text_threshold
        self.detector_processor = AutoProcessor.from_pretrained(detector_dir, local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_dir, local_files_only=True).to(device).eval()
        self.segmenter_processor = Sam2Processor.from_pretrained(segmenter_dir, local_files_only=True)
        self.segmenter = Sam2Model.from_pretrained(segmenter_dir, local_files_only=True).to(device).eval()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=hand_threshold,
            min_hand_presence_confidence=hand_threshold,
            min_tracking_confidence=hand_threshold,
        )
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        crop_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=hand_threshold,
            min_hand_presence_confidence=hand_threshold,
        )
        self.crop_hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(crop_options)
        self.object_track = ObjectTrack()
        self.last_timestamp_ms = -1
        self.bindings = {"detector": model_binding(detector_dir),
                         "segmenter": model_binding(segmenter_dir),
                         "hand_landmarker": model_binding(hand_model)}

    def close(self):
        self.hand_detector.close()
        self.crop_hand_detector.close()

    def predict(self, rgb: np.ndarray, *, prompt: str, time_seconds: float) -> tuple[dict | None, np.ndarray | None, list[dict]]:
        height, width = rgb.shape[:2]
        inputs = self.detector_processor(images=rgb, text=prompt.strip().rstrip(".") + ".",
                                         return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.detector(**inputs)
        detections = self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.detection_threshold,
            text_threshold=self.text_threshold, target_sizes=[(height, width)])[0]
        raw_boxes = detections["boxes"].detach().cpu().numpy()
        boxes, valid = clip_detection_boxes(raw_boxes, width, height)
        scores = detections["scores"].detach().cpu().numpy()
        raw_boxes, boxes, scores = raw_boxes[valid], boxes[valid], scores[valid]
        index, ambiguous = self.object_track.select(boxes, scores)
        obj, mask = None, None
        if index is not None:
            box = boxes[index].tolist()
            inputs = self.segmenter_processor(images=rgb, input_boxes=[[box]],
                                              return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.segmenter(**inputs, multimask_output=False)
            mask = self.segmenter_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"])[0][0, 0].numpy().astype(bool)
            if np.any(mask):
                obj = {"bbox_xyxy": box, "score": float(scores[index]),
                       "detector_bbox_xyxy": raw_boxes[index].tolist(),
                       "mask_quality_score": float(outputs.iou_scores.flatten()[0]),
                       "identity_ambiguous": ambiguous,
                       "mask_basis": "sam2_visible_image_prediction",
                       "score_basis": "grounding_dino_text_detection_not_contact_probability"}
        if obj is None:
            self.object_track.previous_box = None
        timestamp_ms = int(round(time_seconds * 1000))
        if timestamp_ms <= self.last_timestamp_ms:
            raise ValueError("Hand tracking requires source timestamps at least 1 ms apart")
        self.last_timestamp_ms = timestamp_ms
        crop = (0, 0, width, height)
        if obj is not None and not obj["identity_ambiguous"]:
            crop = object_context_crop(obj["bbox_xyxy"], width, height)
            x0, y0, x1, y1 = crop
            hand_rgb = rgb[y0:y1, x0:x1]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(hand_rgb))
            result = self.crop_hand_detector.detect(image)
        else:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = self.hand_detector.detect_for_video(image, timestamp_ms)
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            x0, y0, x1, y1 = crop
            xy = [[float(p.x * (x1 - x0) + x0), float(p.y * (y1 - y0) + y0)] for p in landmarks]
            hands.append({"side": handedness[0].category_name.lower(),
                          "side_confidence": float(handedness[0].score),
                          "side_basis": "mediapipe_handedness_unreviewed",
                          "landmarks_xy": xy,
                          "landmark_basis": "mediapipe_image_estimate",
                          "image_crop_xyxy": list(crop),
                          "detection_confidence": None})
        return obj, mask, hands


def extract_video_observations(*, video: str | Path, object_prompt: str, output: str | Path,
                               detector_dir: str | Path, segmenter_dir: str | Path,
                               hand_model: str | Path, device="cpu", stride: int = 1,
                               detection_threshold=.25, text_threshold=.20,
                               hand_threshold=.30, ffprobe="ffprobe", backend=None) -> Path:
    """Write full-timeline evidence to a fresh directory, preserving missing detections.

    ``backend`` supports local adapters and offline tests; it must expose predict,
    bindings, close. Default inference requires local assets and cannot call APIs.
    """
    import cv2

    video = Path(video).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if not object_prompt.strip():
        raise ValueError("object_prompt must name the target object")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    timing = read_video_timing(video, ffprobe=ffprobe)
    source_hash = sha256_file(video)
    own_backend = backend is None
    if own_backend:
        backend = LocalPerception(detector_dir=detector_dir, segmenter_dir=segmenter_dir,
                                  hand_model=hand_model, device=device,
                                  detection_threshold=detection_threshold,
                                  text_threshold=text_threshold, hand_threshold=hand_threshold)
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    cap = cv2.VideoCapture(str(video))
    tracks = HandTracks()
    frames = []
    decoded = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not decode {video}")
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = decoded
            decoded += 1
            if frame >= timing["frame_count"]:
                raise ValueError("Decoded more frames than the presentation timestamp table")
            if bgr.shape[:2] != (timing["height"], timing["width"]):
                raise ValueError("Changing video dimensions are unsupported")
            if frame % stride:
                continue
            time_seconds = timing["time_seconds"][frame]
            obj, mask, hands = backend.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                               prompt=object_prompt, time_seconds=time_seconds)
            if obj is not None:
                if mask is None or mask.shape != bgr.shape[:2] or not np.any(mask):
                    raise ValueError("Detected object must have a nonempty full-resolution mask")
                mask_path = Path("masks") / f"{frame:06d}.png"
                if not cv2.imwrite(str(output / mask_path), np.asarray(mask, np.uint8) * 255):
                    raise OSError("Failed to save object mask")
                obj = {**obj, "mask_path": mask_path.as_posix(),
                       "mask_sha256": sha256_file(output / mask_path)}
            frames.append({"source_frame": frame, "time_seconds": time_seconds,
                           "object": obj, "hands": tracks.update(
                               hands, time_seconds, (timing["width"], timing["height"]))})
        if decoded != timing["frame_count"]:
            raise ValueError(f"Decoded {decoded} frames, expected {timing['frame_count']}")
        if sha256_file(video) != source_hash:
            raise ValueError("Source video changed during analysis")
        result = {"schema_version": 1, "status": "UNREVIEWED",
                  "source": {"path": str(video), "sha256": source_hash, **timing},
                  "target": {"prompt": object_prompt, "scope": "single_object_single_person"},
                  "perception": {"backend": "local_grounding_dino_sam2_mediapipe",
                                 "models": backend.bindings, "stride": stride,
                                 "device": device, "detection_threshold": detection_threshold,
                                 "hand_context": "automatic_object_box_context_when_unambiguous_else_full_frame",
                                 "text_threshold": text_threshold, "hand_threshold": hand_threshold,
                                 "coordinates": "source_image_pixels_xy_top_left",
                                 "implementation_sha256": sha256_file(__file__),
                                 "software_versions": {
                                     name: importlib.metadata.version(name)
                                     for name in ("numpy", "torch", "transformers", "mediapipe")
                                     if _package_installed(name)},
                                 "limitations": ["Image overlap is not physical contact",
                                     "Missing landmarks are unknown, not evidence of release",
                                     "Handedness and identity require review under occlusion",
                                     "Object-centered hand crops can omit distant approach hands",
                                     "Masks estimate visible silhouette, not 3D surface or hidden contacts"]},
                  "coverage": {"analyzed_frames": len(frames),
                               "object_frames": sum(f["object"] is not None for f in frames),
                               "hand_frames": sum(bool(f["hands"]) for f in frames),
                               "both_hand_frames": sum(len(f["hands"]) == 2 for f in frames)},
                  "frames": frames}
        path = output / "observations.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return path
    finally:
        cap.release()
        if own_backend:
            backend.close()


def _package_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True
