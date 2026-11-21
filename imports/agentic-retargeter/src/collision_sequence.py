"""Compact, exact-frame Newton playback queries in an external runtime.

Only geometry and explicit transforms are sent to the worker. No animation is
interpolated here, and sampled contact glyphs never stand in for a full audit.
This module is safe to import in Blender (standard library only).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

if __package__:
    from .collision import canonical_snapshot, _number, _rows, validate_report, snapshot_digest
else:
    from collision import canonical_snapshot, _number, _rows, validate_report, snapshot_digest


def _matrix(value, label):
    rows = _rows(value, 4, label)
    if len(rows) != 4 or rows[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"{label} must be affine 4x4")
    a, b, c = [row[:3] for row in rows[:3]]
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if not math.isfinite(determinant) or abs(determinant) < 1e-15:
        raise ValueError(f"{label} must be nonsingular")
    return rows


def canonical_sequence(manifest):
    """Bind fixed local meshes/pairs and every requested exact frame transform."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("sequence schema_version must be 1")
    snapshot = canonical_snapshot(manifest.get("snapshot"))
    names = [mesh["name"] for mesh in snapshot["meshes"]]
    limit = manifest.get("sample_limit_per_pair", 64)
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("sample_limit_per_pair must be an integer in [1, 10000]")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("sequence requires exact frames")
    clean_frames = []
    previous = -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        source_frame = _number(frame.get("source_frame"), "source_frame")
        if source_frame < 0 or source_frame <= previous:
            raise ValueError("sequence frames must be nonnegative and strictly increasing")
        previous = source_frame
        matrices = frame.get("matrices")
        if not isinstance(matrices, dict) or set(matrices) != set(names):
            raise ValueError("each frame must bind every mesh transform exactly once")
        clean_frames.append({"source_frame": source_frame, "matrices": {name: _matrix(matrices[name], f"{name} matrix") for name in names}})
    return {"schema_version": 1, "snapshot": snapshot, "frames": clean_frames, "sample_limit_per_pair": limit}


def sequence_digest(manifest):
    payload = json.dumps(canonical_sequence(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_indices(count, limit):
    """Deterministic evenly spaced contact indices, including both endpoints."""
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [(i * (count - 1)) // (limit - 1) for i in range(limit)]


def _provenance(report, snapshot):
    # Reuse the same strict provenance contract as single-frame evidence,
    # without walking the large base geometry again for every playback frame.
    empty = {"schema_version": 1, "status": "diagnostic", "source_frame": snapshot["source_frame"],
             "input_digest": snapshot_digest(snapshot), "units": "m", "backend": report.get("backend"),
             "pairs": [{"mesh_a": a, "mesh_b": b, "contact_count": 0, "points_a_world": [], "points_b_world": [],
                        "point_distances_m": [], "max_point_distance_m": None, "normals_world": [], "signed_normal_gaps_m": []}
                       for a, b in snapshot["pairs"]]}
    validate_report(empty, snapshot)
    for name in ("sequence_client_sha256", "sequence_worker_sha256"):
        if not isinstance(report["backend"].get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", report["backend"][name]):
            raise ValueError(f"missing sequence provenance: {name}")


def validate_sequence_report(report, manifest):
    manifest = canonical_sequence(manifest)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1 or report.get("kind") != "newton_collision_sequence" or report.get("status") != "diagnostic":
        raise ValueError("invalid sequence report schema/status")
    if report.get("units") != "m" or report.get("input_digest") != sequence_digest(manifest):
        raise ValueError("sequence report does not match the requested sequence")
    _provenance(report, manifest["snapshot"])
    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != len(manifest["frames"]):
        raise ValueError("sequence report frame coverage does not match")
    expected_pairs = [tuple(pair) for pair in manifest["snapshot"]["pairs"]]
    for frame, requested in zip(frames, manifest["frames"]):
        if not isinstance(frame, dict) or _number(frame.get("source_frame"), "source_frame") != requested["source_frame"]:
            raise ValueError("sequence report frame order/timing does not match")
        pairs = frame.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
            raise ValueError("sequence report pair coverage does not match")
        for pair, expected in zip(pairs, expected_pairs):
            if not isinstance(pair, dict) or (pair.get("mesh_a"), pair.get("mesh_b")) != expected:
                raise ValueError("sequence report pair order/identity does not match")
            count, size = pair.get("contact_count"), pair.get("sample_count")
            if type(count) is not int or count < 0 or type(size) is not int or size != min(count, manifest["sample_limit_per_pair"]):
                raise ValueError("invalid sequence contact/sample count")
            indices = pair.get("sample_indices")
            if not isinstance(indices, list) or any(type(i) is not int for i in indices) or indices != sample_indices(count, manifest["sample_limit_per_pair"]):
                raise ValueError("sequence contact sample indices do not match")
            a, b = _rows(pair.get("points_a_world"), 3, "points_a_world"), _rows(pair.get("points_b_world"), 3, "points_b_world")
            distances, normals, gaps = pair.get("point_distances_m"), pair.get("normals_world"), pair.get("signed_normal_gaps_m")
            buffers = (a, b, distances, normals, gaps)
            if any(not isinstance(x, list) or len(x) != size for x in buffers):
                raise ValueError("sequence contact buffers have wrong lengths")
            for p0, p1, distance, normal, gap in zip(*buffers):
                distance = _number(distance, "point distance")
                if distance < 0 or not math.isclose(distance, math.dist(p0, p1), abs_tol=2e-6, rel_tol=2e-5):
                    raise ValueError("sequence point distance does not match points")
                if normal is None:
                    if gap is not None:
                        raise ValueError("undefined normal requires null signed gap")
                else:
                    normal = _rows([normal], 3, "normal")[0]
                    if not math.isclose(math.sqrt(sum(n*n for n in normal)), 1.0, abs_tol=2e-5):
                        raise ValueError("sequence normal must be a unit vector or null")
                    expected_gap = sum((x-y)*n for x, y, n in zip(p0, p1, normal))
                    if not math.isclose(_number(gap, "signed gap"), expected_gap, abs_tol=2e-6, rel_tol=2e-5):
                        raise ValueError("sequence signed gap does not match points and normal")
            maximum = pair.get("max_point_distance_m")
            if count == 0:
                if maximum is not None:
                    raise ValueError("empty sequence contact maximum must be null")
            elif _number(maximum, "maximum point distance") < max(distances) - 2e-6:
                raise ValueError("full contact maximum is smaller than a sample")
            elif count == size and not math.isclose(maximum, max(distances), abs_tol=2e-6, rel_tol=2e-5):
                raise ValueError("sequence contact maximum does not match complete samples")
    return report


def run_sequence_query(manifest, runtime, output_dir):
    """One bounded process reuses Newton meshes across all requested frames."""
    manifest = canonical_sequence(manifest)
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    python, repo = Path(runtime.get("python", "")).expanduser(), Path(runtime.get("repo", "")).expanduser()
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("runtime python must be an absolute executable path")
    if not repo.is_absolute() or not (repo / "ho_contact/collision.py").is_file():
        raise ValueError("runtime repo must be an absolute ho_contact checkout")
    device = runtime.get("device", "cpu")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("device must be cpu or cuda:N")
    timeout = _number(runtime.get("sequence_timeout_s", runtime.get("timeout_s", 120)), "timeout_s")
    if timeout <= 0:
        raise ValueError("timeout_s must be positive")
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    source, destination = output / "sequence.json", output / "report.json"
    source.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n")
    command = [str(python), str(Path(__file__).with_name("collision_sequence_backend.py").resolve()), "--input", str(source), "--output", str(destination), "--repo", str(repo.resolve()), "--device", device]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"], env["PYTHONDONTWRITEBYTECODE"] = "1", "1"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, env=env, cwd=output, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"collision sequence timed out after {timeout:g}s; logs: {output}") from error
    if process.returncode:
        raise RuntimeError(f"collision sequence failed (exit {process.returncode}); logs: {output}")
    try:
        report = json.loads(destination.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"collision sequence produced no valid report; logs: {output}") from error
    return validate_sequence_report(report, manifest)
