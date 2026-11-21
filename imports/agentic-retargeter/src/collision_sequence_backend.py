"""Pinned Newton batch worker; fixed topology is reused across exact frames."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys
import time

if __package__:
    from .collision import HO_CONTACT_COMMIT, NEWTON_COMMIT, NORMAL_CONVENTION, POINT_DISTANCE_SEMANTICS, SIGNED_NORMAL_GAP_SEMANTICS
    from .collision_backend import _verify_checkout, _extract_world_normals
    from .collision_sequence import canonical_sequence, sequence_digest, sample_indices, validate_sequence_report
else:
    from collision import HO_CONTACT_COMMIT, NEWTON_COMMIT, NORMAL_CONVENTION, POINT_DISTANCE_SEMANTICS, SIGNED_NORMAL_GAP_SEMANTICS
    from collision_backend import _verify_checkout, _extract_world_normals
    from collision_sequence import canonical_sequence, sequence_digest, sample_indices, validate_sequence_report


def _runtime(repo, device):
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("device must be cpu or cuda:N")
    repo = Path(repo).resolve()
    newton_repo = repo / "third_party/newton"
    ho_hash = _verify_checkout(repo, HO_CONTACT_COMMIT, ["ho_contact", "pyproject.toml", ".gitmodules"])
    newton_hash = _verify_checkout(newton_repo, NEWTON_COMMIT, ["newton", "pyproject.toml"])
    sys.path[:0] = [str(repo), str(newton_repo)]
    import numpy as np
    import warp as wp
    if device == "cpu":
        wp.config.enable_cuda = False
    wp.init()
    import newton
    from ho_contact import collision as upstream
    if not Path(upstream.__file__).resolve().is_relative_to(repo / "ho_contact") or not Path(newton.__file__).resolve().is_relative_to(newton_repo / "newton"):
        raise ValueError("dependencies did not import from the pinned checkout")
    margin = inspect.signature(newton.Model.collide).parameters["rigid_contact_margin"].default
    if margin != 0.01:
        raise ValueError("unexpected Newton mesh contact margin")
    source_dir = Path(__file__).parent
    backend = {
        "name": "ho_contact/Newton mesh contact detection", "ho_contact_commit": HO_CONTACT_COMMIT,
        "newton_commit": NEWTON_COMMIT, "ho_contact_source_sha256": ho_hash,
        "newton_source_sha256": newton_hash, "python_version": sys.version.split()[0],
        "warp_version": importlib.metadata.version("warp-lang"), "numpy_version": np.__version__,
        "device": device, "rigid_contact_margin_m": margin,
        "adapter_client_sha256": hashlib.sha256((source_dir / "collision.py").read_bytes()).hexdigest(),
        "adapter_worker_sha256": hashlib.sha256((source_dir / "collision_backend.py").read_bytes()).hexdigest(),
        "sequence_client_sha256": hashlib.sha256((source_dir / "collision_sequence.py").read_bytes()).hexdigest(),
        "sequence_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "point_distance_semantics": POINT_DISTANCE_SEMANTICS, "normal_convention": NORMAL_CONVENTION,
        "signed_normal_gap_semantics": SIGNED_NORMAL_GAP_SEMANTICS,
        "reuse": "one fixed-topology ContactScene per explicit pair; world vertices refitted for every exact requested frame",
    }
    return np, wp, upstream, backend


def _world_vertices(np, vertices, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(world).all() or np.any(np.abs(world) > np.finfo(np.float32).max):
        raise ValueError("sequence transformed vertices exceed the float32 geometry range")
    return world.astype(np.float32)


def _compact_contacts(np, upstream, contacts, model, state, name_a, name_b, limit):
    count = int(contacts.rigid_contact_count.numpy()[0])
    capacity = int(contacts.rigid_contact_max)
    if count >= capacity:
        raise RuntimeError(f"contact buffer saturated for {name_a}/{name_b}: {count}/{capacity}")
    extracted = upstream.extract_human_object_contacts(contacts, model, state)
    if extracted is None:
        pa, pb, distances = np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    else:
        pa, pb, distances = extracted.human_points, extracted.object_points, extracted.penetration_depths
    normals = _extract_world_normals(contacts, model, len(distances))
    indices = sample_indices(len(distances), limit)
    selected_a, selected_b = pa[indices].tolist(), pb[indices].tolist()
    selected_normals = [normals[i] for i in indices]
    gaps = [None if normal is None else float(np.dot(np.asarray(p0) - p1, normal))
            for p0, p1, normal in zip(selected_a, selected_b, selected_normals)]
    return {"mesh_a": name_a, "mesh_b": name_b, "contact_count": len(distances), "sample_count": len(indices),
            "sample_indices": indices, "points_a_world": selected_a, "points_b_world": selected_b,
            "normals_world": selected_normals, "point_distances_m": distances[indices].tolist(),
            "signed_normal_gaps_m": gaps, "max_point_distance_m": float(distances.max()) if len(distances) else None}


def query_sequence(manifest, repo, device="cpu", progress=None):
    started = time.perf_counter()
    manifest = canonical_sequence(manifest)
    np, wp, upstream, backend = _runtime(repo, device)
    loaded = time.perf_counter()
    meshes = {mesh["name"]: mesh for mesh in manifest["snapshot"]["meshes"]}
    vertices = {name: np.asarray(mesh["vertices"], dtype=np.float64) for name, mesh in meshes.items()}
    triangles = {name: np.asarray(mesh["triangles"], dtype=np.int32) for name, mesh in meshes.items()}
    scenes, frames = {}, []
    identity = np.eye(4, dtype=np.float32)
    build_seconds = 0.0
    with wp.ScopedDevice(device):
        for frame_index, frame in enumerate(manifest["frames"]):
            frame_started = time.perf_counter()
            world = {name: _world_vertices(np, vertices[name], frame["matrices"][name]) for name in meshes}
            pairs = []
            for name_a, name_b in manifest["snapshot"]["pairs"]:
                key = (name_a, name_b)
                if key not in scenes:
                    t0 = time.perf_counter()
                    scenes[key] = upstream.ContactScene(world[name_a], triangles[name_a],
                        [{"vertices": world[name_b], "triangles": triangles[name_b]}], [name_b], [identity])
                    build_seconds += time.perf_counter() - t0
                scene = scenes[key]
                # The pinned class casts local object vertices to float32 before
                # transforming them. Refresh its private local buffer with the
                # float64-transformed world vertices and use identity, matching
                # the single-frame adapter's world-baked arithmetic exactly.
                # Topology is unchanged; collide copies/refits both Warp meshes.
                scene._obj_local_verts[0] = world[name_b]
                contacts, model, state = scene.collide(world[name_a], [identity])
                if model._collision_pipeline.rigid_contact_margin != backend["rigid_contact_margin_m"]:
                    raise ValueError("actual Newton contact margin differs from the recorded default")
                pairs.append(_compact_contacts(np, upstream, contacts, model, state, name_a, name_b, manifest["sample_limit_per_pair"]))
            frames.append({"source_frame": frame["source_frame"], "pairs": pairs})
            if progress:
                progress({"frame_index": frame_index, "frame_count": len(manifest["frames"]), "source_frame": frame["source_frame"],
                          "contact_count": sum(pair["contact_count"] for pair in pairs), "elapsed_s": time.perf_counter() - frame_started})
    report = {"schema_version": 1, "kind": "newton_collision_sequence", "status": "diagnostic", "units": "m",
              "input_digest": sequence_digest(manifest), "backend": backend, "frames": frames,
              "timing": {"runtime_setup_s": loaded - started, "scene_build_s": build_seconds, "total_s": time.perf_counter() - started},
              "limitations": [
                  "Playback glyphs are deterministic samples; full contact counts and maxima are retained, not full contact buffers.",
                  "Only exact requested frames and listed original-mesh pairs were queried; no interpolated-frame or continuous collision guarantee.",
                  "Vertex-to-surface queries within a 0.01 m margin can miss edge/face intersections and deep penetration.",
                  "Normals are Newton query directions mesh_b to mesh_a, not guaranteed outward normals; undefined normals remain null.",
                  "Unsigned paired-point distance and signed normal gap are diagnostics, not penetration depths or forces.",
                  "This cache detects candidates only; no native motion edits, dynamics or motion acceptance are implied.",
              ]}
    return validate_sequence_report(report, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    def progress(record):
        print(json.dumps({"progress": record}, allow_nan=False), flush=True)
    report = query_sequence(json.loads(Path(args.input).read_text()), args.repo, args.device, progress)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
