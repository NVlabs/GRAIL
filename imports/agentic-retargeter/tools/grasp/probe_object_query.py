"""Read-only original-object query probe; no fitting or motion acceptance."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()

import numpy as np

from agentic_retargeter.grasp_io import GraspInputs
from agentic_retargeter.repair_tensor import WarpSignedDistance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing bank/hold JSON with inputs")
    parser.add_argument("--out", type=Path, required=True, help="New probe JSON path")
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-excluded", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    config = json.loads(args.config.read_text())
    inputs = dict(config["inputs"], object_degenerate_policy="exclude_zero_area_queries")
    directory = args.config.resolve().parent
    paths = [(directory / inputs[k]).resolve(strict=True)
             for k in ("native_npz", "model_xml", "object_mesh_npz")]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): sha(p) for p in paths}
    ctx = GraspInputs(inputs, directory)
    evidence = ctx.object_query_provenance
    ids = ctx.object_query_face_ids
    omitted = evidence["excluded_original_zero_area_face_ids"]
    checks = {
        "expected_exclusion_count": len(omitted) == args.expected_excluded,
        "all_query_faces_preserve_original_indices": np.array_equal(ctx.object_query.faces, ctx.object.faces[ids]),
        "all_original_vertices_preserved": np.array_equal(ctx.object_query.vertices, ctx.object.vertices),
        "all_contact_regions_disjoint_from_exclusions": all(
            not set(hand["region_faces"]).intersection(omitted) for hand in ctx.hands.values()),
    }
    if not all(checks.values()):
        raise ValueError("Original geometry/region prerequisite failed: " + repr(checks))
    query = WarpSignedDistance(ctx.object_query.vertices, ctx.object_query.faces, device=args.device)
    # Probe only declared valid material facets; do not infer grasp clearance.
    chosen = np.unique([hand["region_faces"][0] for hand in ctx.hands.values()])
    points = ctx.object.triangles[chosen].mean(axis=1) + .001 * ctx.object.normals[chosen]
    result = query.query(query.torch.tensor(points, dtype=query.torch.float32, device=query.device))
    faces = result["face_ids"].detach().cpu().numpy()
    found = result["found"].detach().cpu().numpy()
    finite = np.isfinite(result["distances"].detach().cpu().numpy())
    checks["probe_points_found_with_finite_distance"] = bool(found.all() and finite.all())
    checks["returned_face_ids_in_query_map"] = bool(((faces >= 0) & (faces < len(ids))).all())
    original_ids = ids[faces] if checks["returned_face_ids_in_query_map"] else np.empty(0, dtype=int)
    checks["returned_triangles_equal_original_material_faces"] = bool(
        checks["returned_face_ids_in_query_map"] and
        np.array_equal(ctx.object_query.faces[faces], ctx.object.faces[original_ids]))
    checks["bound_input_files_unchanged"] = all(sha(p) == before[str(p)] for p in paths)
    report = dict(status="QUERY_PROBE_PASSED" if all(checks.values()) else "QUERY_PROBE_FAILED",
                  checks={k: bool(v) for k, v in checks.items()},
                  input_sha256=before, config_sha256=sha(args.config),
                  device=args.device, original_query_provenance=evidence,
                  warp_query_metadata=query.metadata(),
                  returned_query_face_ids=faces.tolist(), returned_original_face_ids=original_ids.tolist(),
                  configured_probe_original_face_ids=chosen.tolist(),
                  motion_accepted=False, collision_clearance="UNKNOWN",
                  scope="Actual query runtime and immutable geometry/material mapping only; no optimization or full collision audit.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(args.out),
                          original_faces=len(ctx.object.faces), excluded_faces=len(omitted), device=args.device)))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
