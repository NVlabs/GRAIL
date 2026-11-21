"""Compute BPS (Basis Point Set) encoding for object USD meshes.

For each USD file in --object_usd_dir, extracts the mesh, samples surface points,
and computes a fixed-dimensional BPS encoding. Saves one <stem>.npy per object
(plus _basis.npy for reproducibility) in the output directory — mirroring the
structure of object_usd directories.

BPS encoding (Prokudin et al., 2019):
  - Sample N_surface points uniformly from the object mesh surface
  - Center at centroid, normalize to unit bounding sphere
  - For each of d basis points (on Fibonacci sphere), find nearest surface point
  - BPS vector = d scalar distances (shape (d,))

Requires gearenv python (pxr for USD reading):
  python -m grail.retargeting.compute_bps \\
      --object_usd_dir data/motion_lib_genhoi/<dataset>/object_usd \\
      --output_dir data/motion_lib_genhoi/<dataset>/bps \\
      --num_basis 10
"""

import argparse
import os
import sys

import numpy as np  # type: ignore[import]

# ── BPS basis points ──────────────────────────────────────────────────────────


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n evenly-spaced points on a unit sphere via Fibonacci lattice."""
    golden = (1 + 5**0.5) / 2
    i = np.arange(n, dtype=float)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)  # (n, 3)


# ── Mesh extraction from USD ───────────────────────────────────────────────────


def extract_mesh_from_usd(usd_path: str):
    """Extract all vertices and triangle faces from a USD file.

    Handles both all-triangle meshes and mixed (quad/n-gon) meshes by
    triangulating with a simple fan algorithm.

    Returns:
        vertices: (V, 3) float32
        faces: (F, 3) int32, triangulated
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    all_verts = []
    all_faces = []
    vert_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()

        if pts is None or fvc is None or fvi is None:
            continue

        verts = np.array(pts, dtype=np.float32)  # (V, 3)
        counts = np.array(fvc, dtype=np.int32)
        indices = np.array(fvi, dtype=np.int32)

        # Triangulate: fan from first vertex of each polygon
        tris = []
        idx = 0
        for n in counts:
            face_verts = indices[idx : idx + n]
            for j in range(1, n - 1):
                tris.append([face_verts[0], face_verts[j], face_verts[j + 1]])
            idx += n

        if not tris:
            continue

        all_verts.append(verts)
        all_faces.append(np.array(tris, dtype=np.int32) + vert_offset)
        vert_offset += len(verts)

    if not all_verts:
        return None, None

    vertices = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return vertices, faces


# ── Surface point sampling ────────────────────────────────────────────────────


def sample_surface_points(
    vertices: np.ndarray, faces: np.ndarray, n_samples: int = 2048
) -> np.ndarray:
    """Uniformly sample points on the mesh surface (area-weighted).

    Returns:
        pts: (n_samples, 3) float32
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle areas via cross product
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=-1) * 0.5
    areas = np.maximum(areas, 1e-12)

    # Sample triangles proportional to area
    probs = areas / areas.sum()
    tri_ids = np.random.choice(len(faces), size=n_samples, p=probs)

    # Barycentric sampling
    r1 = np.random.rand(n_samples, 1)
    r2 = np.random.rand(n_samples, 1)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    pts = u * v0[tri_ids] + v * v1[tri_ids] + w * v2[tri_ids]
    return pts.astype(np.float32)


# ── BPS computation ───────────────────────────────────────────────────────────


def compute_bps(pts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Compute BPS encoding for a point cloud.

    Args:
        pts: (N, 3) surface points, already centered+normalized
        basis: (d, 3) fixed basis points on unit sphere

    Returns:
        bps: (d,) float32 — L2 distance from each basis point to nearest surface pt
    """
    # (d, N) pairwise distances
    diffs = basis[:, None, :] - pts[None, :, :]  # (d, N, 3)
    dists = np.linalg.norm(diffs, axis=-1)  # (d, N)
    return dists.min(axis=1).astype(np.float32)  # (d,)


def normalize_points(pts: np.ndarray):
    """Center at centroid, scale to unit bounding sphere."""
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.linalg.norm(pts, axis=-1).max()
    if scale > 1e-6:
        pts = pts / scale
    return pts, centroid, scale


# ── Per-USD processing ────────────────────────────────────────────────────────


def process_usd(usd_path: str, basis: np.ndarray, n_surface: int = 2048):
    """Extract, sample, normalize, and encode a single USD file.

    Returns:
        bps: (d,) float32, or None on failure
        error: str or None
    """
    try:
        vertices, faces = extract_mesh_from_usd(usd_path)
    except Exception as e:
        return None, f"USD load error: {e}"

    if vertices is None or len(vertices) == 0:
        return None, "no mesh geometry found"

    try:
        # For very small meshes, use all vertices directly
        if len(vertices) <= n_surface:
            pts = vertices.astype(np.float32)
        else:
            pts = sample_surface_points(vertices, faces, n_samples=n_surface)
    except Exception as e:
        return None, f"sampling error: {e}"

    pts, _, _ = normalize_points(pts)
    bps = compute_bps(pts, basis)
    return bps, None


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute BPS encoding for object USD meshes")
    parser.add_argument("--object_usd_dir", required=True, help="Directory containing .usd files")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory (one .npy per object + _basis.npy)"
    )
    parser.add_argument(
        "--num_basis", type=int, default=10, help="Number of basis points (default: 10)"
    )
    parser.add_argument(
        "--n_surface",
        type=int,
        default=2048,
        help="Surface sample count per object (default: 2048)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    usd_dir = args.object_usd_dir
    if not os.path.isdir(usd_dir):
        print(f"[ERROR] object_usd_dir not found: {usd_dir}")
        sys.exit(1)

    usd_files = sorted(f for f in os.listdir(usd_dir) if f.endswith(".usd") or f.endswith(".usda"))
    if not usd_files:
        print(f"[ERROR] No .usd/.usda files in {usd_dir}")
        sys.exit(1)

    print(f"[BPS] Found {len(usd_files)} USD files in {usd_dir}")
    print(f"[BPS] num_basis={args.num_basis}, n_surface={args.n_surface}, seed={args.seed}")
    print(f"[BPS] Output dir: {args.output_dir}")

    basis = fibonacci_sphere(args.num_basis).astype(np.float32)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save basis points for reproducibility
    np.save(os.path.join(args.output_dir, "_basis.npy"), basis)

    failed = []
    n_saved = 0

    for i, fname in enumerate(usd_files):
        stem = os.path.splitext(fname)[0]
        usd_path = os.path.join(usd_dir, fname)
        bps, err = process_usd(usd_path, basis, n_surface=args.n_surface)

        if err is not None:
            print(f"  [{i+1}/{len(usd_files)}] SKIP {stem}: {err}")
            failed.append((stem, err))
        else:
            np.save(os.path.join(args.output_dir, f"{stem}.npy"), bps)
            n_saved += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(usd_files):
                print(
                    f"  [{i+1}/{len(usd_files)}] Done. bps range [{bps.min():.3f}, {bps.max():.3f}]"
                )

    print(f"\n[BPS] Saved {n_saved} objects → {args.output_dir}/")
    if failed:
        print(
            f"[BPS] {len(failed)} failed: {[s for s, _ in failed[:5]]}{'...' if len(failed) > 5 else ''}"
        )


if __name__ == "__main__":
    main()
