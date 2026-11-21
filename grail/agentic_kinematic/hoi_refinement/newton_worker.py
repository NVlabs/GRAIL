"""Standalone worker for the pinned ho_contact Newton implementation.

Executed in an external Python runtime; never imports into Blender.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from .newton import (
        HO_CONTACT_COMMIT,
        NEWTON_COMMIT,
        POINT_DISTANCE_SEMANTICS,
        canonical_snapshot,
        snapshot_digest,
        validate_report,
    )
else:
    # Never occupy sys.modules['newton']: that name belongs to the external
    # Newton dependency loaded by query(), not our local adapter.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_grail_ho_contact_adapter", Path(__file__).with_name("newton.py")
    )
    _client = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_client)
    HO_CONTACT_COMMIT = _client.HO_CONTACT_COMMIT
    NEWTON_COMMIT = _client.NEWTON_COMMIT
    POINT_DISTANCE_SEMANTICS = _client.POINT_DISTANCE_SEMANTICS
    canonical_snapshot = _client.canonical_snapshot
    snapshot_digest = _client.snapshot_digest
    validate_report = _client.validate_report


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _verify_checkout(repo, expected, paths):
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
        raise ValueError(f"expected a standalone checkout: {repo}")
    if _git(repo, "rev-parse", "HEAD") != expected:
        raise ValueError(f"dependency must be pinned to {expected}: {repo}")
    if _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise ValueError(f"modified dependency implementation: {repo}")
    files = _git(repo, "ls-files", "--", *paths).splitlines()
    digest = hashlib.sha256()
    for filename in sorted(files):
        path = repo / filename
        if path.is_file():
            digest.update(filename.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def query(snapshot, repo, device):
    snapshot = canonical_snapshot(snapshot)
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("device must be cpu or cuda:N")
    repo = Path(repo).resolve()
    newton_repo = repo / "third_party" / "newton"
    ho_hash = _verify_checkout(
        repo, HO_CONTACT_COMMIT, ["ho_contact", "pyproject.toml", ".gitmodules"]
    )
    newton_hash = _verify_checkout(newton_repo, NEWTON_COMMIT, ["newton", "pyproject.toml"])
    sys.path[:0] = [str(repo), str(newton_repo)]
    import numpy as np
    import warp as wp

    if device == "cpu":
        wp.config.enable_cuda = False
    wp.init()
    import newton
    from ho_contact import collision as upstream

    if not Path(upstream.__file__).resolve().is_relative_to(repo / "ho_contact") or not Path(
        newton.__file__
    ).resolve().is_relative_to(newton_repo / "newton"):
        raise ValueError("dependencies did not import from the pinned checkout")
    margin = inspect.signature(newton.Model.collide).parameters["rigid_contact_margin"].default
    if margin != 0.01:
        raise ValueError("unexpected Newton mesh contact margin")
    backend = {
        "name": "ho_contact/Newton mesh contact detection",
        "ho_contact_commit": HO_CONTACT_COMMIT,
        "newton_commit": NEWTON_COMMIT,
        "ho_contact_source_sha256": ho_hash,
        "newton_source_sha256": newton_hash,
        "python_version": sys.version.split()[0],
        "warp_version": importlib.metadata.version("warp-lang"),
        "numpy_version": np.__version__,
        "device": device,
        "rigid_contact_margin_m": margin,
        "adapter_client_sha256": hashlib.sha256(
            Path(__file__).with_name("newton.py").read_bytes()
        ).hexdigest(),
        "adapter_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "point_distance_semantics": POINT_DISTANCE_SEMANTICS,
    }
    meshes = {mesh["name"]: mesh for mesh in snapshot["meshes"]}
    results = []
    with wp.ScopedDevice(device):
        for name_a, name_b in snapshot["pairs"]:
            a, b = meshes[name_a], meshes[name_b]
            # Transform in float64 before the detector's float32 conversion.
            # This avoids overflow in intermediate affine products that cancel.
            va, vb = np.asarray(a["vertices"], dtype=np.float64), np.asarray(
                b["vertices"], dtype=np.float64
            )
            xa, xb = np.asarray(a["matrix_world"], dtype=np.float64), np.asarray(
                b["matrix_world"], dtype=np.float64
            )
            va = (va @ xa[:3, :3].T + xa[:3, 3]).astype(np.float32)
            vb = (vb @ xb[:3, :3].T + xb[:3, 3]).astype(np.float32)
            xb = np.eye(4, dtype=np.float32)
            scene = upstream.ContactScene(
                va,
                np.asarray(a["triangles"], dtype=np.int32),
                [{"vertices": vb, "triangles": np.asarray(b["triangles"], dtype=np.int32)}],
                [name_b],
                [xb],
            )
            contacts, model, state = scene.collide(va, [xb])
            if model._collision_pipeline.rigid_contact_margin != margin:
                raise ValueError("actual Newton contact margin differs from the recorded default")
            count = int(contacts.rigid_contact_count.numpy()[0])
            capacity = int(contacts.rigid_contact_max)
            if count >= capacity:
                raise RuntimeError(
                    f"contact buffer saturated for {name_a}/{name_b}: {count}/{capacity}"
                )
            extracted = upstream.extract_human_object_contacts(contacts, model, state)
            if extracted is None:
                points_a, points_b, distances = [], [], []
            else:
                points_a, points_b = (
                    extracted.human_points.tolist(),
                    extracted.object_points.tolist(),
                )
                distances = extracted.penetration_depths.tolist()
            results.append(
                {
                    "mesh_a": name_a,
                    "mesh_b": name_b,
                    "contact_count": len(distances),
                    "points_a_world": points_a,
                    "points_b_world": points_b,
                    "point_distances_m": distances,
                    "max_point_distance_m": max(distances) if distances else None,
                }
            )
    report = {
        "schema_version": 1,
        "status": "diagnostic",
        "source_frame": snapshot["source_frame"],
        "input_digest": snapshot_digest(snapshot),
        "units": "m",
        "backend": backend,
        "pairs": results,
        "limitations": [
            "Detection only: no joint optimization, motion repair, force or dynamics solver was run.",
            "Mesh contacts are vertex-to-surface queries within a 0.01 m margin, not exhaustive triangle intersection tests.",
            "No contacts does not establish clearance: edge/face intersections and deep penetration may be missed.",
            "Unsigned paired-point distance is not penetration depth or a signed distance field.",
            "Mesh orientation, open surfaces, tessellation and topology affect candidate detection.",
            "Only listed mesh pairs at this snapshot were queried; no whole-scene or temporal acceptance is implied.",
            "Default ground-plane candidates are excluded by the upstream extractor.",
        ],
    }
    return validate_report(report, snapshot)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    report = query(json.loads(Path(args.input).read_text()), args.repo, args.device)
    output = Path(args.output)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
