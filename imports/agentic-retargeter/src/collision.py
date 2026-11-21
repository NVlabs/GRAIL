"""Version-bound, external-runtime Newton contact queries (no motion edits).

This module is safe to import in Blender: Newton and Warp are only imported by
the separate worker. Contact candidates are not a clearance certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

HO_CONTACT_COMMIT = "c3109bc1ec7f22d2e013402650637a728d5c6e50"
NEWTON_COMMIT = "e5a7cfa02c190eec0f6e94c3844b115b5870792a"
POINT_DISTANCE_SEMANTICS = "unsigned norm of paired surface-query points; not penetration depth"
NORMAL_CONVENTION = "Newton world-space query normal oriented mesh_b to mesh_a; null when undefined"
SIGNED_NORMAL_GAP_SEMANTICS = "dot(points_a_world - points_b_world, normals_world); query diagnostic, not penetration depth"
_FLOAT32_MAX = 3.4028234663852886e38


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _rows(value, width, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(f"{label} must have {width} columns")
        numbers = [_number(x, label) for x in row]
        if any(abs(x) > _FLOAT32_MAX for x in numbers):
            raise ValueError(f"{label} exceeds the float32 geometry range")
        rows.append(numbers)
    return rows


def canonical_snapshot(snapshot):
    """Validate a snapshot and return its deterministic, JSON-safe payload.

    Meshes contain local vertices in metres and affine matrix_world transforms
    whose translations are also metres. Blender unit conversion belongs to the
    exporter. Explicit named pairs determine exactly which meshes are queried.
    Extra UI metadata is excluded from the geometry digest.
    """
    if not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("snapshot schema_version must be 1")
    if snapshot.get("units", "m") != "m":
        raise ValueError("snapshot units must be m")
    frame = _number(snapshot.get("source_frame"), "source_frame")
    if frame < 0:
        raise ValueError("source_frame must be nonnegative")
    meshes = snapshot.get("meshes")
    if not isinstance(meshes, list) or len(meshes) < 2:
        raise ValueError("snapshot needs at least two meshes")
    canonical = []
    names = set()
    for mesh in meshes:
        if not isinstance(mesh, dict):
            raise ValueError("each mesh must be an object")
        name = mesh.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("mesh names must be nonempty and unique")
        names.add(name)
        vertices = _rows(mesh.get("vertices"), 3, f"{name} vertices")
        if len(vertices) < 3:
            raise ValueError(f"{name} needs at least three vertices")
        triangles = mesh.get("triangles")
        if not isinstance(triangles, list) or not triangles:
            raise ValueError(f"{name} needs triangles")
        for triangle in triangles:
            if not isinstance(triangle, (list, tuple)) or len(triangle) != 3 or any(type(i) is not int or i < 0 or i >= len(vertices) for i in triangle) or len(set(triangle)) != 3:
                raise ValueError(f"{name} triangle indices are invalid")
        matrix = _rows(mesh.get("matrix_world"), 4, f"{name} matrix_world")
        if len(matrix) != 4 or matrix[3] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError(f"{name} matrix_world must be affine 4x4")
        a, b, c = [row[:3] for row in matrix[:3]]
        det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if not math.isfinite(det) or abs(det) < 1e-15:
            raise ValueError(f"{name} matrix_world must be nonsingular")
        for vertex in vertices:
            for row in matrix[:3]:
                world = sum(row[i] * vertex[i] for i in range(3)) + row[3]
                if not math.isfinite(world) or abs(world) > _FLOAT32_MAX:
                    raise ValueError(f"{name} transformed vertices exceed the float32 geometry range")
        canonical.append({"name": name, "vertices": vertices, "triangles": [list(t) for t in triangles], "matrix_world": matrix})
    pairs = snapshot.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("snapshot needs explicit mesh pairs")
    unique = set()
    clean_pairs = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(x, str) or x not in names for x in pair) or pair[0] == pair[1]:
            raise ValueError("each pair must name two distinct existing meshes")
        unordered = tuple(sorted(pair))
        if unordered in unique:
            raise ValueError("duplicate mesh pair")
        unique.add(unordered)
        clean_pairs.append(list(pair))
    return {"schema_version": 1, "source_frame": frame, "units": "m", "meshes": sorted(canonical, key=lambda m: m["name"]), "pairs": sorted(clean_pairs)}


def snapshot_digest(snapshot):
    payload = json.dumps(canonical_snapshot(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_report(report, snapshot):
    """Reject stale, malformed, unbound or unsupported contact evidence."""
    snapshot = canonical_snapshot(snapshot)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1 or report.get("status") != "diagnostic":
        raise ValueError("invalid collision report schema/status")
    frame = _number(report.get("source_frame"), "report source_frame")
    if report.get("input_digest") != snapshot_digest(snapshot) or frame != snapshot["source_frame"] or report.get("units") != "m":
        raise ValueError("collision report does not match the current snapshot")
    backend = report.get("backend", {})
    if not isinstance(backend, dict) or backend.get("ho_contact_commit") != HO_CONTACT_COMMIT or backend.get("newton_commit") != NEWTON_COMMIT:
        raise ValueError("collision report has unsupported dependency provenance")
    for field in ("ho_contact_source_sha256", "newton_source_sha256", "adapter_client_sha256", "adapter_worker_sha256"):
        value = backend.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"collision report has invalid provenance: {field}")
    device = backend.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("collision report has invalid device provenance")
    if _number(backend.get("rigid_contact_margin_m"), "rigid_contact_margin_m") != 0.01:
        raise ValueError("collision report has an unsupported contact margin")
    for field in ("python_version", "warp_version", "numpy_version"):
        value = backend.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", value):
            raise ValueError(f"collision report has invalid runtime provenance: {field}")
    if backend.get("point_distance_semantics") != POINT_DISTANCE_SEMANTICS:
        raise ValueError("collision report has unsupported point distance semantics")
    has_normals = "normal_convention" in backend or "signed_normal_gap_semantics" in backend
    if has_normals and (backend.get("normal_convention") != NORMAL_CONVENTION or backend.get("signed_normal_gap_semantics") != SIGNED_NORMAL_GAP_SEMANTICS):
        raise ValueError("collision report has unsupported normal convention or signed gap semantics")
    pairs = report.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(snapshot["pairs"]):
        raise ValueError("collision report pair coverage does not match")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("invalid report pair")
        key = (pair.get("mesh_a"), pair.get("mesh_b"))
        if key not in map(tuple, snapshot["pairs"]) or key in seen:
            raise ValueError("collision report contains an unexpected or duplicate pair")
        seen.add(key)
        count = pair.get("contact_count")
        if type(count) is not int or count < 0:
            raise ValueError("invalid contact_count")
        a = _rows(pair.get("points_a_world"), 3, "points_a_world")
        b = _rows(pair.get("points_b_world"), 3, "points_b_world")
        distances = pair.get("point_distances_m")
        if not isinstance(distances, list) or len(a) != count or len(b) != count or len(distances) != count:
            raise ValueError("contact buffer lengths do not match contact_count")
        for p0, p1, distance in zip(a, b, distances):
            distance = _number(distance, "point_distances_m")
            expected = math.dist(p0, p1)
            if distance < 0 or not math.isclose(distance, expected, abs_tol=2e-6, rel_tol=2e-5):
                raise ValueError("contact point distance does not match returned points")
        # Schema 1 reports made before normal extraction remain valid evidence,
        # but absent normals are unavailable, never reconstructed from points.
        has_pair_normals = "normals_world" in pair or "signed_normal_gaps_m" in pair
        if has_pair_normals != has_normals:
            raise ValueError("contact normals require consistent buffers and backend semantics")
        if has_normals:
            normals = pair.get("normals_world")
            gaps = pair.get("signed_normal_gaps_m")
            if not isinstance(normals, list) or not isinstance(gaps, list) or len(normals) != count or len(gaps) != count:
                raise ValueError("normal buffer lengths do not match contact_count")
            for p0, p1, normal, gap in zip(a, b, normals, gaps):
                if normal is None:
                    if gap is not None:
                        raise ValueError("undefined normal must have a null signed gap")
                    continue
                normal = _rows([normal], 3, "normals_world")[0]
                if not math.isclose(math.sqrt(sum(n * n for n in normal)), 1.0, abs_tol=2e-5):
                    raise ValueError("contact normals must be unit vectors or null")
                expected = sum((x - y) * n for x, y, n in zip(p0, p1, normal))
                if not math.isclose(_number(gap, "signed_normal_gaps_m"), expected, abs_tol=2e-6, rel_tol=2e-5):
                    raise ValueError("signed normal gap does not match points and normal")
        maximum = pair.get("max_point_distance_m")
        if count:
            if not math.isclose(_number(maximum, "max_point_distance_m"), max(distances), abs_tol=2e-6, rel_tol=2e-5):
                raise ValueError("invalid max_point_distance_m")
        elif maximum is not None:
            raise ValueError("empty contact maximum must be null")
    return report


def run_query(snapshot, runtime, output_dir):
    """Run one bounded, foreground query; persist evidence in a new directory.

    runtime = {python: executable, repo: pinned checkout, device: cpu|cuda:N,
               timeout_s: 120}. Failure logs remain beside snapshot.json.
    """
    snapshot = canonical_snapshot(snapshot)
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    python = Path(runtime.get("python", "")).expanduser()
    repo = Path(runtime.get("repo", "")).expanduser()
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("runtime python must be an absolute executable path")
    if not repo.is_absolute() or not (repo / "ho_contact" / "collision.py").is_file():
        raise ValueError("runtime repo must be an absolute ho_contact checkout")
    device = runtime.get("device", "cpu")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("device must be cpu or cuda:N")
    timeout = _number(runtime.get("timeout_s", 120), "timeout_s")
    if timeout <= 0:
        raise ValueError("timeout_s must be positive")
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "snapshot.json"
    source.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    destination = output / "report.json"
    command = [str(python), str(Path(__file__).with_name("collision_backend.py").resolve()), "--input", str(source), "--output", str(destination), "--repo", str(repo.resolve()), "--device", device]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, env=env, cwd=output, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"collision query timed out after {timeout:g}s; logs: {output}") from error
    if process.returncode:
        raise RuntimeError(f"collision query failed (exit {process.returncode}); logs: {output}")
    try:
        report = json.loads(destination.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"collision query produced no valid report; logs: {output}") from error
    return validate_report(report, snapshot)
