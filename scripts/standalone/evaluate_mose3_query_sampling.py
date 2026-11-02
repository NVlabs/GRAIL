"""Replay fixed CAD-PnP with first-frame-only subsets of cached MoSE3 tracks.

This does not run MoSE3 inference. The research directory supplies the frozen
bridge and manifest; outputs must go to a different directory. No invalid poses
are filled and no future pose or mask participates in selecting queries.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from grail.adapters.mose3_sampling import query_sampling_diagnostics, select_query_indices


def bind(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    root, output = args.research_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Outputs must be outside the frozen research directory")
    sys.dont_write_bytecode = True
    bridge_path = root / "scripts/mose3_pose_bridge.py"
    spec = importlib.util.spec_from_file_location("sampling_frozen_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    manifest = root / "runs/mose3_full121_756_all100_v001/collection_manifest.json"
    collection = json.loads(manifest.read_text())
    cases_manifest = root / "benchmark/cases_manifest.json"
    cases = {r["case_id"]: r for r in json.loads(cases_manifest.read_text())["records"]}
    records = [r for r in collection["records"] if args.case is None or r["case_id"] in args.case]
    if args.case and set(args.case) != {r["case_id"] for r in records}:
        raise ValueError("Unknown case identifier")
    code = [
        bind(__file__),
        bind(bridge_path),
        bind(manifest),
        bind(cases_manifest),
        bind(sys.modules[select_query_indices.__module__].__file__),
    ]
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for record in records:
        cid = record["case_id"]
        case = cases[cid]
        source = Path(record["source_directory"])
        a_path, tracks_path = source / "anchors.npz", source / "window_000_tracks.npz"
        anchors = np.load(a_path, allow_pickle=False)
        tracks = np.load(tracks_path, allow_pickle=False)
        frame_ids = tracks["frame_ids"]
        if not np.array_equal(frame_ids, np.arange(case["frames"])):
            raise ValueError(f"{cid}: cached window is not the complete source timeline")
        # Cast before projection, matching the original bridge's float64 path.
        pixels, positive = bridge.project(tracks["tracks_local"], tracks["model_K"])
        pixels = bridge.map_pixels(pixels, np.linalg.inv(anchors["native_to_network"]))
        visibility = np.where(positive, tracks["visibility_logits"], -np.inf)
        eligible = np.isfinite(anchors["object_points"]).all(1) & positive[0]
        bindings = code + [bind(a_path), bind(tracks_path), bind(source / "record.json")]
        if bindings[-1]["sha256"] != record["source_record_sha256"]:
            raise ValueError("Frozen source record changed")
        source_pose = Path(record["branches"]["pnp"]["source_path"])
        if bind(source_pose)["sha256"] != record["branches"]["pnp"]["sha256"]:
            raise ValueError("Frozen baseline pose changed")
        baseline = np.load(source_pose, allow_pickle=False)
        for method in [
            "baseline2048",
            "farthest128",
            "farthest256",
            "raster_uniform128",
            "raster_uniform256",
        ]:
            dest = output / "runs" / cid / method
            if dest.exists():
                raise FileExistsError(f"Preserve previous result: {dest}")
            dest.mkdir(parents=True)
            start = time.monotonic()
            row = {
                "case_id": cid,
                "method": method,
                "inputs_and_code": bindings,
                "baseline_pose": bind(source_pose),
                "fps": case["fps"],
                "frame_count": len(frame_ids),
                "future_data_for_selection": False,
                "sampling_pool": "cached original2048; no new dense forward",
                "smoothing": False,
                "invalid_fill": False,
            }
            try:
                if method == "baseline2048":
                    indices = np.arange(len(anchors["pixels_xy"]))
                else:
                    count = 128 if method.endswith("128") else 256
                    strategy = method[:-3]
                    indices = select_query_indices(
                        anchors["pixels_xy"],
                        visibility[0],
                        count,
                        method=strategy,
                        eligible_mask=eligible,
                    )
                np.save(dest / "selected_indices.npy", indices)
                row["selected_count"] = len(indices)
                if method != "baseline2048":
                    row["sampling"] = query_sampling_diagnostics(
                        anchors["pixels_xy"],
                        visibility[0],
                        indices,
                        eligible_mask=eligible,
                        object_points=anchors["object_points"],
                    )
                result = bridge.pnp_pose_sequence(
                    anchors["object_points"][indices],
                    pixels[:, indices],
                    anchors["K"],
                    anchors["T_camera_object_initial"],
                    visibility[:, indices],
                    native_hw=(case["height"], case["width"]),
                )
                if method == "baseline2048":
                    row["baseline_validity_exact"] = bool(
                        np.array_equal(result["valid"], baseline["valid"])
                    )
                    v = result["valid"] & baseline["valid"]
                    row["baseline_pose_max_abs_difference"] = float(
                        np.max(
                            np.abs(result["T_camera_object"][v] - baseline["T_camera_object"][v])
                        )
                    )
                    if (
                        not row["baseline_validity_exact"]
                        or row["baseline_pose_max_abs_difference"] > 1e-8
                    ):
                        raise ValueError(
                            "Baseline replay mismatch; do not compare sampling changes"
                        )
                np.savez_compressed(
                    dest / "poses.npz",
                    frame_ids=frame_ids,
                    T_camera_object=result["T_camera_object"],
                    valid=result["valid"],
                )
                row.update(
                    status="complete",
                    valid_frames=int(result["valid"].sum()),
                    per_frame=result["per_frame"],
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                poses = np.full((len(frame_ids), 4, 4), np.nan)
                poses[0] = anchors["T_camera_object_initial"]
                valid = np.zeros(len(frame_ids), bool)
                valid[0] = True
                np.savez_compressed(
                    dest / "poses.npz", frame_ids=frame_ids, T_camera_object=poses, valid=valid
                )
                if method == "baseline2048":
                    dump(dest / "record.json", row)
                    raise
            row["elapsed_seconds"] = time.monotonic() - start
            dump(dest / "record.json", row)
            summary.append(
                {k: row[k] for k in ["case_id", "method", "status", "elapsed_seconds"]}
                | {"valid_frames": row.get("valid_frames", 1)}
            )
            print(json.dumps(summary[-1]), flush=True)
        label = (
            "all100"
            if args.case is None
            else hashlib.sha256("\n".join(args.case).encode()).hexdigest()[:12]
        )
        dump(output / f"summary_{label}.json", summary)


if __name__ == "__main__":
    main()
