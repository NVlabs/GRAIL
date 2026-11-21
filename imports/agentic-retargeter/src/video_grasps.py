"""Source-video grasp hypotheses and key selection, before native robot editing.

These are image-space observations, not contact ground truth. In particular,
projected overlap and co-motion cannot determine depth or force closure.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


class GraspPlanError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Settings:
    near_hand_scale: float = 0.20
    closure_threshold: float = 0.18
    opposition_hand_scale: float = 0.65
    stable_seconds: float = 0.12
    max_gap_seconds: float = 0.12
    approach_seconds: float = 0.20
    object_motion_fraction: float = 0.05
    max_keyframes: int = 32

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise GraspPlanError(f"{key} must be positive and finite")
        if type(self.max_keyframes) is not int or self.max_keyframes < 6:
            raise GraspPlanError("max_keyframes must be an integer >= 6")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_observations(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise GraspPlanError("Expected observations schema_version=1")
    source = data.get("source", {})
    for name in ("width", "height", "frame_count"):
        if type(source.get(name)) is not int or source[name] <= 0:
            raise GraspPlanError(f"source.{name} must be a positive integer")
    if not _finite(source.get("fps")) or source["fps"] <= 0:
        raise GraspPlanError("source.fps must be positive")
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise GraspPlanError("Source video SHA256 is required")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise GraspPlanError("No decoded frames")
    timestamps = source.get("time_seconds")
    if timestamps is not None:
        if (not isinstance(timestamps, list) or len(timestamps) != source["frame_count"]
                or not all(_finite(t) and t >= 0 for t in timestamps)
                or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
            raise GraspPlanError("Invalid source presentation timestamp table")
    previous_frame, previous_time = -1, -1.
    for frame in frames:
        index, time = frame.get("source_frame"), frame.get("time_seconds")
        if type(index) is not int or not previous_frame < index < source["frame_count"]:
            raise GraspPlanError("Frame indices must increase within the source video")
        if not _finite(time) or not previous_time < time or time < 0:
            raise GraspPlanError("Source timestamps must be finite and strictly increasing")
        previous_frame, previous_time = index, time
        if timestamps is not None and abs(time - timestamps[index]) > 1e-6:
            raise GraspPlanError("Frame time disagrees with source presentation timestamp")
        obj = frame.get("object")
        if obj is not None:
            box = np.asarray(obj.get("bbox_xyxy"), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise GraspPlanError("Invalid object bounding box")
            if (box[0] < 0 or box[1] < 0 or box[2] > source["width"] or box[3] > source["height"]):
                raise GraspPlanError("Object box outside image")
            if not isinstance(obj.get("mask_path"), str) or not obj["mask_path"]:
                raise GraspPlanError("An object segmentation mask is required; a box is not a surface")
        hands = frame.get("hands", [])
        if not isinstance(hands, list):
            raise GraspPlanError("hands must be a list")
        ids = set()
        for hand in hands:
            track = hand.get("track_id")
            if not isinstance(track, str) or not track or track in ids:
                raise GraspPlanError("Each hand needs a unique nonempty track_id per frame")
            ids.add(track)
            points = np.asarray(hand.get("landmarks_xy"), dtype=float)
            if points.shape != (21, 2) or not np.isfinite(points).all():
                raise GraspPlanError("Expected 21 finite pixel-space hand landmarks")
            if hand.get("side", "unknown") not in ("left", "right", "unknown"):
                raise GraspPlanError("Hand side must be left, right or unknown")
    return data


def hand_features(hand, mask, settings):
    """Projected proximity to a segmented silhouette plus finger-shape cues."""
    points = np.asarray(hand["landmarks_xy"], dtype=float)
    scale = float(np.linalg.norm(points[0] - points[9]))
    if scale < 2:
        return {"valid": False, "reason": "hand_too_small_or_degenerate"}
    inside = np.asarray(mask, dtype=bool)
    if inside.ndim != 2 or not inside.any():
        return {"valid": False, "reason": "empty_object_mask"}
    # Use original mask boundary, not its bounding box or a convex hull.
    boundary = inside & ~binary_erosion(inside)
    distance, indices = distance_transform_edt(~boundary, return_indices=True)
    outside_distance = distance_transform_edt(~inside)
    tips = (4, 8, 12, 16, 20)
    anchors, proximities = [], []
    for landmark in tips:
        point = points[landmark]
        x, y = np.rint(point).astype(int)
        if not (0 <= x < inside.shape[1] and 0 <= y < inside.shape[0]):
            continue
        gap = float(outside_distance[y, x]) / scale
        proximities.append(gap)
        if gap <= settings.near_hand_scale:
            anchors.append({"hand_landmark": landmark, "hand_xy_px": point.tolist(),
                            "object_silhouette_xy_px": [int(indices[1, y, x]), int(indices[0, y, x])],
                            "projected_overlap": bool(inside[y, x]),
                            "silhouette_distance_px": float(distance[y, x]),
                            "outside_gap_hand_scale": gap})
    curvature = []
    for start in (5, 9, 13, 17):
        chain = points[start:start + 4]
        length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
        if length > 1:
            curvature.append(1 - float(np.linalg.norm(chain[-1] - chain[0])) / length)
    closure = float(np.median(curvature)) if curvature else 0.
    opposition = float(np.linalg.norm(points[4] - points[8])) / scale
    near = len(anchors) >= 2
    shape_support = closure >= settings.closure_threshold or opposition <= settings.opposition_hand_scale
    return {"valid": True, "hand_scale_px": scale, "closure_cue": closure,
            "opposition_hand_scale": opposition, "near": near,
            "shape_support": shape_support, "supported": near and shape_support,
            "anchors": anchors, "palm_xy_px": points[[0, 5, 9, 13, 17]].mean(axis=0).tolist(),
            "handedness": hand.get("side", "unknown"),
            "handedness_score": hand.get("side_confidence"),
            "heuristic_support": min(1., len(anchors) / 3) * (0.65 if shape_support else 0.25)}


def plan(data, mask_loader, settings=None):
    """Plan source-frame hypotheses without writing, editing or selecting motion.

    mask_loader(frame) returns a bool HxW array. It is injectable for tests;
    production run() binds every referenced mask and the original source video.
    """
    data = validate_observations(data)
    settings = settings or Settings()
    settings.validate()
    frames, source = data["frames"], data["source"]
    tracks = sorted({h["track_id"] for f in frames for h in f.get("hands", [])})
    features = {track: [] for track in tracks}
    for frame in frames:
        obj = frame.get("object")
        mask = None if obj is None or obj.get("identity_ambiguous", False) else np.asarray(mask_loader(frame), dtype=bool)
        if mask is not None and mask.shape != (source["height"], source["width"]):
            raise GraspPlanError("Segmentation mask must have source image dimensions")
        hand_map = {h["track_id"]: h for h in frame.get("hands", [])}
        for track in tracks:
            feature = {"valid": False, "reason": "hand_or_object_not_observed"}
            if mask is not None and track in hand_map:
                feature = hand_features(hand_map[track], mask, settings)
            features[track].append(feature)
    # Supporting co-motion requires visible object movement. A stationary hand
    # next to a stationary object adds no co-motion evidence.
    for track in tracks:
        for i in range(1, len(frames)):
            previous, current = features[track][i - 1:i + 1]
            dt = frames[i]["time_seconds"] - frames[i - 1]["time_seconds"]
            if not previous["valid"] or not current["valid"] or dt > settings.max_gap_seconds:
                continue
            centers = [np.asarray(f["object"]["bbox_xyxy"], float).reshape(2, 2).mean(axis=0) for f in frames[i - 1:i + 1]]
            object_delta = centers[1] - centers[0]
            hand_delta = np.asarray(current["palm_xy_px"]) - previous["palm_xy_px"]
            scale = current["hand_scale_px"]
            moving = np.linalg.norm(object_delta) / scale / dt > 0.10
            follows = np.linalg.norm(hand_delta - object_delta) / scale / dt < 0.60
            current["co_motion_support"] = bool(moving and follows and previous["near"] and current["near"])
            if current["co_motion_support"]:
                current["supported"] = True
                current["heuristic_support"] = max(current["heuristic_support"], .75)

    events, keys, contacts, unknown = [], {}, [], []
    # Per-frame handedness can flip during occlusion. Such a track is useful
    # temporally but must not select a left/right native robot joint group.
    sides = {}
    for track, values in features.items():
        observed = {v["handedness"] for v in values if v["valid"] and v["handedness"] != "unknown"}
        sides[track] = next(iter(observed)) if len(observed) == 1 else "unknown"
    if not tracks:
        unknown.append({"track_id": None, "first_frame": frames[0]["source_frame"],
                        "last_frame": frames[-1]["source_frame"], "reason": "no_hands_observed"})
    for previous, current in zip(frames, frames[1:]):
        if current["time_seconds"] - previous["time_seconds"] > settings.max_gap_seconds:
            unknown.append({"track_id": None, "frame_before": previous["source_frame"],
                            "frame_after": current["source_frame"], "reason": "sampling_gap"})
    def add_key(index, role, track):
        source_frame = frames[index]["source_frame"]
        key = keys.setdefault(source_frame, {"source_frame": source_frame,
                              "time_seconds": frames[index]["time_seconds"], "roles": [], "tracks": []})
        if role not in key["roles"]:
            key["roles"].append(role)
        if track is not None and track not in key["tracks"]:
            key["tracks"].append(track)

    # Preserve overview context even when no grasp is reliably observed. This
    # does not convert missing hand evidence into a negative-contact label.
    add_key(0, "clip_start_context", None)
    add_key(len(frames) - 1, "clip_end_context", None)
    context_events = []
    object_indices = [i for i, f in enumerate(frames) if f.get("object") is not None
                      and not f["object"].get("identity_ambiguous", False)]
    if object_indices:
        baseline = object_indices[0]
        box = np.asarray(frames[baseline]["object"]["bbox_xyxy"], float).reshape(2, 2)
        center, diagonal = box.mean(axis=0), np.linalg.norm(box[1] - box[0])
        onset = previous = None
        for j in object_indices[1:]:
            current_box = np.asarray(frames[j]["object"]["bbox_xyxy"], float).reshape(2, 2)
            if np.linalg.norm(current_box.mean(axis=0) - center) < settings.object_motion_fraction * diagonal:
                onset = previous = None
                continue
            if previous is None or j != previous + 1 or frames[j]["time_seconds"] - frames[previous]["time_seconds"] > settings.max_gap_seconds:
                onset = j
            previous = j
            if frames[j]["time_seconds"] - frames[onset]["time_seconds"] >= settings.stable_seconds:
                add_key(onset, "object_motion_context", None)
                before = max((k for k in range(onset) if frames[k]["time_seconds"] <=
                              frames[onset]["time_seconds"] - settings.approach_seconds), default=0)
                add_key(before, "before_object_motion_context", None)
                context_events.append({"type": "object_image_displacement_candidate",
                                       "source_frame": frames[onset]["source_frame"],
                                       "baseline_source_frame": frames[baseline]["source_frame"],
                                       "confirmed_frame": frames[j]["source_frame"],
                                       "status": "UNREVIEWED"})
                break

    for track, values in features.items():
        i = 0
        while i < len(values):
            if not values[i]["valid"]:
                start = i
                while i + 1 < len(values) and not values[i + 1]["valid"]:
                    i += 1
                unknown.append({"track_id": track, "first_frame": frames[start]["source_frame"],
                                "last_frame": frames[i]["source_frame"], "reason": "missing_image_evidence"})
                i += 1
                continue
            if not values[i]["supported"]:
                i += 1
                continue
            start = end = i
            while (end + 1 < len(values) and values[end + 1].get("supported", False)
                   and frames[end + 1]["time_seconds"] - frames[end]["time_seconds"] <= settings.max_gap_seconds):
                end += 1
            i = end + 1
            duration = frames[end]["time_seconds"] - frames[start]["time_seconds"]
            if duration < settings.stable_seconds:
                continue
            onset = start
            while (onset > 0 and values[onset - 1].get("near", False)
                   and frames[onset]["time_seconds"] - frames[onset - 1]["time_seconds"] <= settings.max_gap_seconds):
                onset -= 1
            confirmed = next(j for j in range(start, end + 1)
                             if frames[j]["time_seconds"] - frames[start]["time_seconds"] >= settings.stable_seconds)
            lower = onset - 1 if (onset > 0 and values[onset - 1]["valid"] and
                                 frames[onset]["time_seconds"] - frames[onset - 1]["time_seconds"] <= settings.max_gap_seconds) else None
            event = {"id": f"grasp_{len(events):03d}", "type": "grasp_candidate", "track_id": track,
                     "side": sides[track], "handedness_status": "UNREVIEWED",
                     "first_near_frame": frames[onset]["source_frame"],
                     "onset_bracket_frames": [None if lower is None else frames[lower]["source_frame"], frames[onset]["source_frame"]],
                     "first_supported_frame": frames[start]["source_frame"],
                     "confirmed_frame": frames[confirmed]["source_frame"], "last_supported_frame": frames[end]["source_frame"],
                     "observed_duration_seconds": duration,
                     "end_reason": "clip_end" if end + 1 == len(values) else (
                         "evidence_missing" if not values[end + 1]["valid"] else (
                             "sampling_gap" if frames[end + 1]["time_seconds"] - frames[end]["time_seconds"] > settings.max_gap_seconds
                             else "support_cue_ended")),
                     "release_confirmed": False, "status": "UNREVIEWED"}
            events.append(event)
            add_key(onset, "first_near_candidate", track)
            add_key(confirmed, "grasp_confirmation_candidate", track)
            add_key((start + end) // 2, "hold_candidate", track)
            add_key(end, "last_supported", track)
            before = [j for j in range(onset) if values[j]["valid"] and
                      frames[onset]["time_seconds"] - frames[j]["time_seconds"] <= settings.approach_seconds]
            if before:
                add_key(before[0], "pre_candidate_context", track)
            # This is only image motion: never label it lift-off in 3D.
            initial_box = np.asarray(frames[onset]["object"]["bbox_xyxy"], float).reshape(2, 2)
            center = initial_box.mean(axis=0)
            diagonal = np.linalg.norm(initial_box[1] - initial_box[0])
            for j in range(onset + 1, end + 1):
                box = np.asarray(frames[j]["object"]["bbox_xyxy"], float).reshape(2, 2)
                if np.linalg.norm(box.mean(axis=0) - center) >= settings.object_motion_fraction * diagonal:
                    event["object_image_motion_frame"] = frames[j]["source_frame"]
                    add_key(j, "object_image_motion_candidate", track)
                    break
    # No automatic truncation of event boundaries: caller must split long clips.
    if len(keys) > settings.max_keyframes:
        raise GraspPlanError(f"{len(keys)} selected frames exceed max_keyframes={settings.max_keyframes}; split the clip or increase the limit")
    for index, frame in enumerate(frames):
        if frame["source_frame"] not in keys:
            continue
        for track in keys[frame["source_frame"]]["tracks"]:
            feature = features[track][index]
            if feature.get("near"):
                contacts.append({"source_frame": frame["source_frame"], "track_id": track,
                                 "side": sides[track], "observed_side": feature["handedness"],
                                 "handedness_status": "UNREVIEWED", "coordinate_space": "source_image_pixels",
                                 "kind": "projected_silhouette_hypothesis", "status": "UNREVIEWED",
                                 "heuristic_support": feature["heuristic_support"],
                                 "anchors": feature["anchors"], "object_local_3d": None})
    return {"schema_version": 1, "status": "UNREVIEWED", "source": source,
            "target": data.get("target", {}), "settings": asdict(settings),
            "event_semantics": "Supported evidence windows; fragmentation does not count distinct physical grasps.",
            "events": events, "context_events": context_events, "keyframes": [keys[k] for k in sorted(keys)],
            "contact_candidates": contacts, "unknown_intervals": unknown,
            "coverage": {"decoded_frames": len(frames), "source_frame_count": source["frame_count"],
                         "hand_observed_frames": {t: sum(v["valid"] for v in values) for t, values in features.items()}},
            "limitations": ["Heuristic image evidence, not a calibrated contact probability or ground truth.",
                            "Silhouette anchors cannot determine hidden contact, object-local 3D position, normals or force closure.",
                            "Missing observations do not mean release. Object image motion does not establish lift-off.",
                            "Review handedness, occlusion, source timing and native robot feasibility before editing."]}


def run(observations, output, settings=None, *, render=True):
    """Bind source/masks, write a new plan and optional source keyframe preview."""
    from PIL import Image

    observation_path = Path(observations).expanduser().resolve(strict=True)
    data = validate_observations(json.loads(observation_path.read_text()))
    source = Path(data["source"]["path"]).expanduser()
    if not source.is_absolute():
        source = observation_path.parent / source
    source = source.resolve(strict=True)
    if sha256(source) != data["source"]["sha256"]:
        raise GraspPlanError("Source video changed since detection")
    from .video_perception import read_video_timing

    timing = read_video_timing(source)
    if any(timing[k] != data["source"][k] for k in ("frame_count", "width", "height")):
        raise GraspPlanError("Source video dimensions/frame count disagree with observations")
    if abs(timing["fps"] - data["source"]["fps"]) > 1e-5:
        raise GraspPlanError("Source video frame rate disagrees with observations")
    for frame in data["frames"]:
        if abs(frame["time_seconds"] - timing["time_seconds"][frame["source_frame"]]) > 1e-6:
            raise GraspPlanError("Observation timing disagrees with source video PTS")
    bindings = {}
    def load_mask(frame):
        relative = Path(frame["object"]["mask_path"])
        mask_path = (observation_path.parent / relative).resolve(strict=True)
        if relative.is_absolute() or not mask_path.is_relative_to(observation_path.parent):
            raise GraspPlanError("Mask paths must stay inside the observations directory")
        bindings[str(relative)] = sha256(mask_path)
        expected = frame["object"].get("mask_sha256")
        if expected is not None and bindings[str(relative)] != expected:
            raise GraspPlanError("Segmentation mask changed since detection")
        with Image.open(mask_path) as image:
            return np.asarray(image.convert("L")) > 0
    result = plan(data, load_mask, settings)
    if sha256(source) != data["source"]["sha256"]:
        raise GraspPlanError("Source video changed during planning")
    result["source"] = {**result["source"], "path": str(source)}
    result["provenance"] = {"observations": {"path": str(observation_path), "sha256": sha256(observation_path)},
                            "mask_sha256": bindings, "planner_sha256": sha256(__file__)}
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "grasp_plan.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, field in (("keyframes", "keyframes"), ("contact_candidates", "contact_candidates"), ("grasp_events", "events")):
        (output / f"{name}.json").write_text(json.dumps({"schema_version": 1, "status": "UNREVIEWED", "source": result["source"],
                            "plan_sha256": sha256(result_path), field: result[field]}, indent=2, allow_nan=False) + "\n")
    if render:
        render_keyframes(result, output)
    return {"plan": str(result_path), "evidence_windows": len(result["events"]), "keyframes": len(result["keyframes"]), "status": "UNREVIEWED"}


def render_keyframes(result, output):
    """English contact-candidate sheets, always showing the original image."""
    import cv2
    import textwrap
    from PIL import Image, ImageDraw

    output = Path(output)
    requested = {k["source_frame"]: k for k in result["keyframes"]}
    if not requested:
        return
    capture = cv2.VideoCapture(result["source"]["path"])
    tiles, index = [], 0
    try:
        while True:
            success, bgr = capture.read()
            if not success:
                break
            if index in requested:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                canvas = Image.fromarray(rgb)
                draw = ImageDraw.Draw(canvas)
                for contact in result["contact_candidates"]:
                    if contact["source_frame"] != index:
                        continue
                    for anchor in contact["anchors"]:
                        x, y = anchor["object_silhouette_xy_px"]
                        hx, hy = anchor["hand_xy_px"]
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#76b900", width=3)
                        draw.line((hx, hy, x, y), fill="#ffb347", width=2)
                # A portrait source and a contact crop share one tile; source
                # pixels remain visible rather than replacing them with a rig.
                annotated = canvas.copy()
                canvas.thumbnail((300, 360))
                tile = Image.new("RGB", (640, 414), "#151b24")
                tile.paste(canvas, (0, 54))
                draw = ImageDraw.Draw(tile)
                contact_points = [p for c in result["contact_candidates"] if c["source_frame"] == index
                                  for a in c["anchors"] for p in (a["hand_xy_px"], a["object_silhouette_xy_px"])]
                if contact_points:
                    points = np.asarray(contact_points)
                    low, high = points.min(axis=0), points.max(axis=0)
                    padding = max(35., float(np.max(high - low)) * .35)
                    bounds = (max(0, int(low[0] - padding)), max(0, int(low[1] - padding)),
                              min(annotated.width, int(high[0] + padding)), min(annotated.height, int(high[1] + padding)))
                    crop = annotated.crop(bounds)
                    crop.thumbnail((330, 325))
                    # PIL thumbnail never enlarges; a magnified crop is useful
                    # for small source hands while keeping pixel evidence clear.
                    scale = min(330 / crop.width, 325 / crop.height)
                    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)))
                    tile.paste(crop, (305, 82))
                    draw.text((305, 61), "Projected contact candidates (zoom)", fill="#a8d869")
                else:
                    draw.text((305, 80), "No supported contact anchor at this key", fill="#aaaaaa")
                key = requested[index]
                draw.text((8, 5), f"Frame {index} | {key['time_seconds']:.3f}s | UNREVIEWED 2D candidates", fill="white")
                labels = " | ".join(role.replace("_", " ") for role in key["roles"])
                for line_number, line in enumerate(textwrap.wrap(labels, width=90)[:2]):
                    draw.text((8, 22 + 14 * line_number), line, fill="#a8d869")
                tile.save(output / f"keyframe_{index:06d}.png")
                tiles.append(tile)
            index += 1
    finally:
        capture.release()
    if len(tiles) != len(requested):
        raise GraspPlanError("Could not decode all planned source keyframes")
    for page, offset in enumerate(range(0, len(tiles), 8)):
        group = tiles[offset:offset + 8]
        sheet = Image.new("RGB", (1280, 414 * math.ceil(len(group) / 2)), "#151b24")
        for i, tile in enumerate(group):
            sheet.paste(tile, ((i % 2) * 640, (i // 2) * 414))
        sheet.save(output / f"keyframes_{page:02d}.png")
