"""Reusable CPU diagnostics for native HOI geometry, without fitting or acceptance.

The depth path consumes GRAIL's original-resolution calibrated MoGe tensor list
and source mask cache directly. All candidates use the same source rays. Missing
coverage and wrong-surface hits are retained, including at frame zero. Empty
denominators are null rather than a perfect score. This module never calibrates
depth, edits geometry, infers contact, or interprets an open mesh as a solid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "grail.hoi_diagnostics.v1"


def _image(value):
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError("Depth/mask must have original source [H,W] resolution")
    return value


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _stats(values):
    values = np.asarray(values).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "max": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def temporal_metrics(points, fps):
    """World-space finite differences of unchanged native material points."""
    points = np.asarray(points)
    if points.ndim not in (2, 3) or points.shape[-1] != 3 or not len(points):
        raise ValueError("Temporal points must be nonempty [T,3] or [T,N,3]")
    if not np.isfinite(points).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Temporal points and positive source fps must be finite")
    result = {}
    for order, unit in ((1, "speed_m_s"), (2, "acceleration_m_s2"), (3, "jerk_m_s3")):
        values = np.linalg.norm(np.diff(points.astype(np.float64), n=order, axis=0), axis=-1)
        values *= fps**order
        result[unit] = _stats(values)
        result[unit]["maximum_at_source_frame"] = (
            int(np.unravel_index(np.argmax(values), values.shape)[0] + order)
            if values.size else None
        )
    return result


def geometry_metrics(geometry, fps, *, expected_frames=None):
    """Validate native arrays and report full-timeline motion, not a quality vote."""
    frames = None
    for surface in ("human", "object"):
        vertices = np.asarray(geometry[f"{surface}_vertices"])
        faces = np.asarray(geometry[f"{surface}_faces"])
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or min(vertices.shape[:2]) < 1:
            raise ValueError(f"{surface}_vertices must be nonempty [T,V,3]")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{surface}_vertices contain nonfinite coordinates")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in "iu" or faces.min() < 0
                or faces.max() >= vertices.shape[1]):
            raise ValueError(f"{surface}_faces must index the original vertices")
        if frames is not None and len(vertices) != frames:
            raise ValueError("Human and object timeline lengths differ")
        frames = len(vertices)
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"Geometry has {frames} frames, expected {expected_frames}")
    result = {
        "frame_count": frames,
        "fps": float(fps),
        "fps_source": "explicit caller source rate; geometry export does not store fps",
        "finite_native_vertices": True,
        "original_face_indices_in_bounds": True,
        "human_vertex_count": int(geometry["human_vertices"].shape[1]),
        "object_vertex_count": int(geometry["object_vertices"].shape[1]),
        "object_centroid": temporal_metrics(geometry["object_vertices"].mean(1), fps),
        "object_material_vertices": temporal_metrics(geometry["object_vertices"], fps),
    }
    if "joints" in geometry:
        joints = np.asarray(geometry["joints"])
        if joints.ndim != 3 or joints.shape[0] != frames or joints.shape[-1] != 3:
            raise ValueError("joints must cover the full native timeline as [T,J,3]")
        result["all_joints"] = temporal_metrics(joints, fps)
        result["per_joint"] = {
            str(joint): temporal_metrics(joints[:, joint], fps) for joint in range(joints.shape[1])
        }
    return result


def compare_geometry(reference, candidate):
    """Exact topology/camera equality plus displacement; no inferred frozen pose."""
    fields = ("human_faces", "object_faces", "camera_R", "camera_T", "focal", "image_size")
    equality = {key: np.array_equal(reference[key], candidate[key]) for key in fields}
    displacement = {}
    for key in ("human_vertices", "object_vertices", "joints"):
        if key not in reference or key not in candidate:
            continue
        a, b = reference[key], candidate[key]
        if a.shape != b.shape:
            raise ValueError(f"Cannot compare different native {key} shapes")
        displacement[key] = {
            "exactly_equal": bool(np.array_equal(a, b)),
            "displacement_m": _stats(np.linalg.norm(a - b, axis=-1)),
        }
    return {"exact_equal_fields": equality, "geometry_change": displacement}


def frame_alignment(target_depth, source_masks, predicted_layers):
    """One native source frame; layers map candidate -> human/object camera-z.

    A source-object ray hit by a human is missing object coverage. Its actual
    frontmost depth is also measured separately and the wrong hit is counted.
    Common rays require the intended visible surface in every candidate.
    """
    depth = _image(target_depth)
    masks = {}
    for key, value in source_masks.items():
        value = _image(value)
        if value.shape != depth.shape or not np.isfinite(value).all():
            raise ValueError("Source masks and depth must have identical original resolution")
        masks[key] = value > 0
    if set(masks) != {"human", "object"}:
        raise ValueError("Require both original human and object masks")
    source_overlap = masks["human"] & masks["object"]
    # Same occlusion convention as the previous object-depth review: human wins.
    masks["object"] &= ~masks["human"]
    valid_target = np.isfinite(depth) & (depth > 0)
    visible, front = {}, {}
    for label, layers in predicted_layers.items():
        if set(layers) != {"human", "object"}:
            raise ValueError("Each prediction must contain human and object depth layers")
        h, o = (np.asarray(layers[key]) for key in ("human", "object"))
        if h.shape != depth.shape or o.shape != depth.shape:
            raise ValueError("Rendered and fixed source depth must share original resolution")
        hhit, ohit = np.isfinite(h) & (h > 0), np.isfinite(o) & (o > 0)
        visible[label] = {
            "human": hhit & (~ohit | (h < o)),
            "object": ohit & (~hhit | (o <= h)),
        }
        front[label] = np.minimum(np.where(hhit, h, np.inf), np.where(ohit, o, np.inf))
    output = {label: {} for label in predicted_layers}
    for surface in ("human", "object"):
        fixed = masks[surface] & valid_target
        common = fixed.copy()
        for candidate in visible.values():
            common &= candidate[surface]
        for label, layers in predicted_layers.items():
            predicted = visible[label][surface]
            covered = fixed & predicted
            combined = fixed & np.isfinite(front[label])
            wrong = combined & ~predicted
            union = masks[surface] | predicted
            intersection = masks[surface] & predicted
            record = {
                "source_mask_pixels": int(masks[surface].sum()),
                "source_overlap_pixels_human_priority": int(source_overlap.sum()),
                "invalid_target_depth_pixels": int((masks[surface] & ~valid_target).sum()),
                "target_pixels": int(fixed.sum()),
                "covered_pixels": int(covered.sum()),
                "missing_pixels": int((fixed & ~predicted).sum()),
                "wrong_surface_hit_pixels": int(wrong.sum()),
                "common_pixels": int(common.sum()),
                "silhouette_intersection_pixels": int(intersection.sum()),
                "silhouette_union_pixels": int(union.sum()),
                "silhouette_iou": float(intersection.sum() / union.sum()) if union.any() else None,
            }
            for prefix, selected, prediction in (
                ("covered", covered, layers[surface]),
                ("common", common, layers[surface]),
                ("frontmost_any_surface", combined, front[label]),
            ):
                delta = np.asarray(prediction)[selected].astype(np.float64) - depth[selected]
                record[f"{prefix}_count"] = int(delta.size)
                record[f"{prefix}_absolute_error_sum_m"] = float(np.abs(delta).sum())
                record[f"{prefix}_signed_error_sum_m"] = float(delta.sum())
                record[f"{prefix}_mae_m"] = float(np.abs(delta).mean()) if delta.size else None
                record[f"{prefix}_bias_m"] = float(delta.mean()) if delta.size else None
            output[label][surface] = record
    return output


def _summarize_alignment(rows):
    result = {}
    for surface in ("human", "object"):
        records = [row[surface] for row in rows]
        keys = [key for key in records[0] if key.endswith("_pixels") or key.endswith("_count")]
        summary = {key: sum(row[key] for row in records) for key in keys}
        denominator = summary["silhouette_union_pixels"]
        summary["pixel_weighted_silhouette_iou"] = (
            summary["silhouette_intersection_pixels"] / denominator if denominator else None
        )
        for prefix in ("covered", "common", "frontmost_any_surface"):
            denominator = summary[f"{prefix}_count"]
            for metric, key in (("mae", "absolute"), ("bias", "signed")):
                total = sum(row[f"{prefix}_{key}_error_sum_m"] for row in records)
                summary[f"pixel_weighted_{prefix}_{metric}_m"] = (
                    total / denominator if denominator else None
                )
        result[surface] = summary
    return result


def depth_alignment(geometries, depth, masks, rasterizer):
    from grail.visualization.hoi_review import source_camera

    first = next(iter(geometries.values()))
    frame_count = len(first["human_vertices"])
    height, width = map(int, first["image_size"])
    if len(depth) != frame_count or set(masks) != set(range(frame_count)):
        raise ValueError("Depth and mask caches must cover every source frame including frame 0")
    for geometry in geometries.values():
        if len(geometry["human_vertices"]) != frame_count:
            raise ValueError("Depth candidates have different frame counts")
        for key in ("camera_R", "camera_T", "focal", "image_size"):
            if not np.array_equal(first[key], geometry[key]):
                raise ValueError(f"Candidates have different calibrated {key}")
    result = {label: [] for label in geometries}
    cameras = {label: source_camera(geometry) for label, geometry in geometries.items()}
    for frame in range(frame_count):
        target = _image(depth[frame])
        if target.shape != (height, width):
            raise ValueError("MoGe cache differs from original camera resolution; no resize allowed")
        layers = {}
        for label, geometry in geometries.items():
            rotation, translation = cameras[label]
            focal = np.asarray(geometry["focal"]).reshape(-1)
            if focal.size == 1:
                focal = np.repeat(focal, 2)
            layers[label] = {}
            for surface in ("human", "object"):
                vertices = geometry[f"{surface}_vertices"][frame] @ rotation + translation
                rendered, _ = rasterizer.raster(
                    vertices, geometry[f"{surface}_faces"], focal, height, width,
                    slot=f"diagnostics_{label}_{surface}",
                )
                layers[label][surface] = rendered
        row = frame_alignment(target, {"human": masks[frame][1], "object": masks[frame][0]}, layers)
        for label in result:
            result[label].append({"frame": frame, **row[label]})
    return {
        "status": "PERFORMED",
        "scope": "Original-resolution fixed source rays; original native faces; no recalibration. "
        "Common rays hit the intended visible surface in every listed candidate. "
        "Frontmost-any-surface error must be read with wrong-surface hit counts.",
        "candidate_labels": list(geometries),
        "summary": {label: _summarize_alignment(rows) for label, rows in result.items()},
        "frames": result,
    }


def load_depth_and_masks(depth_path, masks_path, frame_count, metadata_path=None):
    """Only load trusted local GRAIL outputs; mask NPZ uses its historical pickle format."""
    import torch

    depth = torch.load(depth_path, map_location="cpu", weights_only=True)
    with np.load(masks_path, allow_pickle=True) as archive:
        masks = archive["masks"].item()
    metadata_path = Path(metadata_path or f"{depth_path}.json")
    metadata = json.loads(metadata_path.read_text())
    calibration = metadata.get("gt_calibration", {})
    if metadata.get("model", {}).get("depth_unit") != "meter":
        raise ValueError("Depth metadata must explicitly declare meter units")
    if metadata.get("frame_count") != frame_count:
        raise ValueError("Depth metadata frame count differs from native source timeline")
    if (calibration.get("status") != "aligned"
            or calibration.get("method") != "per_frame_background_linear_depth_scale_shift"
            or len(calibration.get("frames", [])) != frame_count
            or not all(calibration.get(key) is True for key in (
                "first_frame_obj_mask_supplied", "first_frame_human_mask_supplied", "video_masks_supplied"))):
        raise ValueError("Depth diagnostics require completed original background-GT calibration")
    for index, frame in enumerate(calibration["frames"]):
        if (frame.get("frame_idx") != index or frame.get("status") != "aligned"
                or frame.get("foreground_excluded") is not True
                or not np.isfinite(frame.get("scale", np.nan)) or frame["scale"] <= 0
                or not np.isfinite(frame.get("shift", np.nan))):
            raise ValueError("Depth metadata has invalid or incomplete background calibration")
    records = {
        "depth": file_record(depth_path), "masks": file_record(masks_path),
        "depth_metadata": file_record(metadata_path), "calibration_method": calibration["method"],
        "target_provenance": "Supplied existing calibrated cache; never fitted to candidate geometry",
    }
    return depth, masks, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--depth-metadata", type=Path)
    parser.add_argument("--raster-cache", type=Path, default=Path(".cache/hoi_review"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.depth) != bool(args.masks):
        parser.error("--depth and --masks are required together")
    if args.depth_metadata and not args.depth:
        parser.error("--depth-metadata requires --depth")
    if args.output.exists():
        raise FileExistsError(f"Preserve existing report: {args.output}")
    from grail.visualization.hoi_review import CpuRasterizer, build_raster, load_geometry

    start = time.perf_counter()
    geometries, inputs = {}, {}
    for argument in args.geometry:
        label, separator, value = argument.partition("=")
        if not separator or not label.strip() or label in geometries:
            parser.error("Each --geometry needs a unique nonempty LABEL=PATH")
        geometries[label] = load_geometry(value)
        inputs[label] = file_record(value)
    report = {
        "schema": SCHEMA, "status": "DIAGNOSTIC_NOT_ACCEPTANCE", "inputs": inputs,
        "geometry": {label: geometry_metrics(g, args.fps, expected_frames=args.expected_frames)
                     for label, g in geometries.items()},
        "contact_and_original_mesh_intersections": {
            "status": "NOT_PERFORMED",
            "reason": "Use existing phase-contact/material-pair and original-mesh review tools separately",
        },
        "source_depth": {"status": "NOT_PERFORMED", "reason": "No fixed depth/masks supplied"},
    }
    first_label, first = next(iter(geometries.items()))
    report["comparison_reference"] = first_label
    report["comparisons"] = {label: compare_geometry(first, g) for label, g in geometries.items()}
    if args.depth:
        depth, masks, records = load_depth_and_masks(
            args.depth, args.masks, len(first["human_vertices"]), args.depth_metadata
        )
        library, receipt = build_raster(args.raster_cache)
        report["fixed_targets"] = records
        report["raster_build"] = receipt
        report["source_depth"] = depth_alignment(geometries, depth, masks, CpuRasterizer(library))
    report["elapsed_seconds"] = time.perf_counter() - start
    report["limitations"] = "No contact correctness, signed penetration, foot stance inference, " \
        "source keypoint error or physical validation is established by these diagnostics."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
