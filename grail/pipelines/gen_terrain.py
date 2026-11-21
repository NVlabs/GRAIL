#!/usr/bin/env python3
"""
Generate synthetic terrain assets (curbs, slopes, stairs) as OBJ + MTL + texture.

Produces ready-to-render terrain meshes directly without intermediate MJCF/USD steps.
Each asset gets a merged OBJ with per-piece colors baked into a texture atlas.

Usage:
    python -m grail.pipelines.gen_terrain --type all --num 300 --output_dir data/Terrain
    python -m grail.pipelines.gen_terrain --type curb --num 40
    python -m grail.pipelines.gen_terrain --type slope --num 100 --seed 1234
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tunables (match original generate_*.py scripts)
# ---------------------------------------------------------------------------

# Sizes are pre-scaled for the G1 character.
# applied `obj_scale: [0.6, 0.6, 0.6]` at render time; that scale is now baked in here
# so the rendered terrain matches G1 directly with `obj_scale: [1.0, 1.0, 1.0]`.

# Curbs
CURB_COUNT_RANGE = (2, 5)
CURB_WIDTH_RANGE = (0.06, 0.30)
CURB_DEPTH_RANGE = (0.60, 0.60)
CURB_HEIGHT_RANGE = (0.03, 0.18)
CURB_GAP_RANGE = (0.03, 1.20)
CURB_Y_OFFSET_RANGE = (-0.12, 0.12)

# Slopes
SLOPE_SEGMENT_COUNT_RANGE = (2, 5)
SLOPE_LENGTH_RANGE = (0.48, 1.44)
SLOPE_WIDTH_RANGE = (1.20, 1.80)
SLOPE_HEIGHT_DELTA_RANGE = (-0.27, 0.27)
SLOPE_MIN_SURFACE_HEIGHT = 0.012

# Stairs
STAIR_STEP_COUNT_RANGE = (3, 8)
STAIR_RISE_RANGE = (0.06, 0.15)
STAIR_TREAD_RANGE = (0.09, 0.24)
STAIR_WIDTH_RANGE = (1.20, 1.80)

TEX_SIZE = 256
EDGE_SPLIT_ANGLE = math.radians(30)

# Base seeds (match originals for reproducibility)
SEED_CURB = 2024
SEED_SLOPE = 6060
SEED_STAIRS = 7070


# ---------------------------------------------------------------------------
# Geometry builders — each returns list of (vertices, faces, color)
# ---------------------------------------------------------------------------


def _rng(low, high):
    return random.uniform(low, high)


def _box_mesh(cx, cy, cz, sx, sy, sz):
    """Return (verts, quads) for an axis-aligned box centered at (cx,cy,cz) with half-sizes (sx,sy,sz)."""
    v = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # front (-y)
        (2, 3, 7, 6),  # back (+y)
        (0, 4, 7, 3),  # left (-x)
        (1, 2, 6, 5),  # right (+x)
    ]
    return v, quads


def _wedge_mesh(x_off, length, width, h0, h1):
    """Return (verts, tris) for a wedge from h0 to h1 along x, extruded along y."""
    hw = width / 2.0
    v = [
        (x_off, -hw, 0.0),
        (x_off + length, -hw, 0.0),
        (x_off, hw, 0.0),
        (x_off + length, hw, 0.0),
        (x_off, -hw, h0),
        (x_off + length, -hw, h1),
        (x_off, hw, h0),
        (x_off + length, hw, h1),
    ]
    tris = [
        (0, 1, 3),
        (0, 3, 2),  # bottom
        (4, 5, 7),
        (4, 7, 6),  # top
        (0, 1, 5),
        (0, 5, 4),  # side y-
        (2, 3, 7),
        (2, 7, 6),  # side y+
        (0, 2, 6),
        (0, 6, 4),  # side x=0
        (1, 3, 7),
        (1, 7, 5),  # side x=length
    ]
    return v, tris


def _random_color():
    return (_rng(0.4, 0.9), _rng(0.4, 0.9), _rng(0.4, 0.9))


def generate_curb_pieces(seed):
    """Generate curb geometry. Returns list of (verts, faces, color, is_quad)."""
    random.seed(seed)
    num = random.randint(*CURB_COUNT_RANGE)
    pieces = []
    cursor_x = 0.0

    for _ in range(num):
        sx = _rng(*CURB_WIDTH_RANGE)
        sy = _rng(*CURB_DEPTH_RANGE)
        sz = _rng(*CURB_HEIGHT_RANGE)
        px = cursor_x + sx
        py = _rng(*CURB_Y_OFFSET_RANGE)
        pz = sz
        verts, quads = _box_mesh(px, py, pz, sx, sy, sz)
        pieces.append((verts, quads, _random_color(), True))
        cursor_x = px + sx + _rng(*CURB_GAP_RANGE)

    return pieces


def generate_slope_pieces(seed):
    """Generate slope geometry. Returns list of (verts, faces, color, is_quad)."""
    random.seed(seed)
    num = random.randint(*SLOPE_SEGMENT_COUNT_RANGE)
    pieces = []
    cursor_x = 0.0
    current_h = SLOPE_MIN_SURFACE_HEIGHT

    for _ in range(num):
        length = _rng(*SLOPE_LENGTH_RANGE)
        width = _rng(*SLOPE_WIDTH_RANGE)
        end_h = max(SLOPE_MIN_SURFACE_HEIGHT, current_h + _rng(*SLOPE_HEIGHT_DELTA_RANGE))
        verts, tris = _wedge_mesh(cursor_x, length, width, current_h, end_h)
        pieces.append((verts, tris, _random_color(), False))
        cursor_x += length
        current_h = end_h

    return pieces


def generate_stair_pieces(
    seed,
    uniform=False,
    step_count_range=None,
    rise_range=None,
    tread_range=None,
    width_range=None,
):
    """Generate stair geometry. Returns list of (verts, faces, color, is_quad)."""
    random.seed(seed)
    step_count_range = step_count_range or STAIR_STEP_COUNT_RANGE
    rise_range = rise_range or STAIR_RISE_RANGE
    tread_range = tread_range or STAIR_TREAD_RANGE
    width_range = width_range or STAIR_WIDTH_RANGE

    num_steps = random.randint(*step_count_range)
    width_y = _rng(*width_range)

    if uniform:
        rise = _rng(*rise_range)
        tread = _rng(*tread_range)
        rises = [rise] * num_steps
        treads_up = [tread] * num_steps
        treads_down = [tread] * num_steps
    else:
        rises = [_rng(*rise_range) for _ in range(num_steps)]
        treads_up = [_rng(*tread_range) for _ in range(num_steps)]
        treads_down = [_rng(*tread_range) for _ in range(num_steps)]

    total_length = sum(treads_up) + sum(treads_down)
    pieces = []

    start_up = 0.0
    start_down_offsets = []
    accum = 0.0
    for t in treads_down:
        start_down_offsets.append(accum)
        accum += t

    for i, rise in enumerate(rises):
        x_min = start_up
        x_max = total_length - start_down_offsets[i]
        length = x_max - x_min
        size_x = length / 2.0
        size_y = width_y / 2.0
        size_z = rise / 2.0
        bottom = sum(rises[:i])
        pos_x = x_min + size_x
        pos_z = bottom + size_z

        verts, quads = _box_mesh(pos_x, 0.0, pos_z, size_x, size_y, size_z)
        pieces.append((verts, quads, _random_color(), True))
        start_up += treads_up[i]

    return pieces


# ---------------------------------------------------------------------------
# OBJ + MTL + Texture output
# ---------------------------------------------------------------------------


def _face_normal(v0, v1, v2):
    """Compute face normal from 3 vertices."""
    a = np.array(v1) - np.array(v0)
    b = np.array(v2) - np.array(v0)
    n = np.cross(a, b)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(n / length)


def _normals_equal(n1, n2, threshold):
    """Check if two normals are within threshold angle."""
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(n1, n2))))
    return math.acos(dot) < threshold


def create_grid_texture(colors, tex_path):
    """Create a texture image with a grid of solid-color tiles."""
    n = len(colors)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), (128, 128, 128))
    tile_w = TEX_SIZE // cols
    tile_h = TEX_SIZE // rows

    for i, (r, g, b) in enumerate(colors):
        col = i % cols
        row = i // cols
        pil_row = (rows - 1) - row
        x0 = col * tile_w
        y0 = pil_row * tile_h
        ri = int(max(0, min(255, r * 255)))
        gi = int(max(0, min(255, g * 255)))
        bi = int(max(0, min(255, b * 255)))
        for y in range(y0, min(y0 + tile_h, TEX_SIZE)):
            for x in range(x0, min(x0 + tile_w, TEX_SIZE)):
                img.putpixel((x, y), (ri, gi, bi))

    img.save(tex_path, quality=95)
    return cols, rows


def write_mtl(mtl_path):
    with open(mtl_path, "w") as f:
        f.write("# Terrain material\n")
        f.write("newmtl terrain_material\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("Ns 10.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write("map_Kd texture.jpg\n")


def write_obj(obj_path, pieces, cols, rows):
    """Write merged OBJ with edge-split vertices (like Blender's Edge Split modifier).

    Vertices are duplicated at sharp edges (>30°) so that Cycles renders
    clean shading without triangle seam artifacts.
    """
    all_verts = []
    all_normals = []
    all_uvs = []
    all_faces = []

    for piece_idx, (verts, faces, color, is_quad) in enumerate(pieces):
        # UV for this piece
        col = piece_idx % cols
        row = piece_idx // cols
        u = (col + 0.5) / cols
        v = (row + 0.5) / rows
        uv_idx = len(all_uvs)
        all_uvs.append((u, v))

        # Compute per-face normals
        face_normals = []
        for face in faces:
            face_normals.append(_face_normal(verts[face[0]], verts[face[1]], verts[face[2]]))

        # Build smooth groups: faces sharing a vertex with similar normals
        # get the same (duplicated) vertex. Different smooth groups at a vertex
        # get separate vertex copies (edge split).
        # Key: (original_vert_idx, smooth_normal_tuple) -> new_vert_idx
        vert_map = {}

        for fi, face in enumerate(faces):
            fn = face_normals[fi]
            for vi in face:
                # Compute smoothed normal at this vertex for this face
                avg = np.array(fn, dtype=float)
                for fi2, face2 in enumerate(faces):
                    if fi2 == fi:
                        continue
                    if vi in face2 and _normals_equal(fn, face_normals[fi2], EDGE_SPLIT_ANGLE):
                        avg += np.array(face_normals[fi2])
                norm = np.linalg.norm(avg)
                if norm > 1e-12:
                    avg = avg / norm
                # Round to avoid floating point key issues
                smooth_key = (vi, round(avg[0], 5), round(avg[1], 5), round(avg[2], 5))

                if smooth_key not in vert_map:
                    new_idx = len(all_verts)
                    all_verts.append(verts[vi])
                    all_normals.append(tuple(avg))
                    vert_map[smooth_key] = new_idx

        # Build faces with new vertex indices
        for fi, face in enumerate(faces):
            fn = face_normals[fi]
            face_data = []
            for vi in face:
                avg = np.array(fn, dtype=float)
                for fi2, face2 in enumerate(faces):
                    if fi2 == fi:
                        continue
                    if vi in face2 and _normals_equal(fn, face_normals[fi2], EDGE_SPLIT_ANGLE):
                        avg += np.array(face_normals[fi2])
                norm = np.linalg.norm(avg)
                if norm > 1e-12:
                    avg = avg / norm
                smooth_key = (vi, round(avg[0], 5), round(avg[1], 5), round(avg[2], 5))
                new_idx = vert_map[smooth_key]
                # OBJ is 1-indexed; vertex and normal share the same index
                face_data.append(f"{new_idx + 1}/{uv_idx + 1}/{new_idx + 1}")
            all_faces.append(face_data)

    # Write OBJ
    lines = ["# Terrain mesh", "mtllib model.mtl", "usemtl terrain_material"]
    for vx, vy, vz in all_verts:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for u, v in all_uvs:
        lines.append(f"vt {u:.6f} {v:.6f}")
    for nx, ny, nz in all_normals:
        lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
    for face_data in all_faces:
        lines.append("f " + " ".join(face_data))

    with open(obj_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_terrain(pieces, output_dir, name):
    """Export a single terrain asset as OBJ + MTL + texture."""
    os.makedirs(output_dir, exist_ok=True)

    obj_path = os.path.join(output_dir, "model.obj")
    mtl_path = os.path.join(output_dir, "model.mtl")
    tex_path = os.path.join(output_dir, "texture.jpg")

    colors = [color for _, _, color, _ in pieces]
    cols, rows = create_grid_texture(colors, tex_path)
    write_mtl(mtl_path)
    write_obj(obj_path, pieces, cols, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = {
    "curb": (generate_curb_pieces, SEED_CURB),
    "slope": (generate_slope_pieces, SEED_SLOPE),
    "stairs": (generate_stair_pieces, SEED_STAIRS),
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic terrain assets")
    parser.add_argument(
        "--type",
        choices=["curb", "slope", "stairs", "all"],
        default="all",
        help="Terrain type to generate",
    )
    parser.add_argument("--num", type=int, default=300, help="Number of assets per type")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed")
    parser.add_argument("--output_dir", type=str, default="data/Terrain", help="Output directory")
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Stairs only: use fixed rise and tread for every step in each staircase",
    )
    parser.add_argument("--stair-min-steps", type=int, default=STAIR_STEP_COUNT_RANGE[0])
    parser.add_argument("--stair-max-steps", type=int, default=STAIR_STEP_COUNT_RANGE[1])
    parser.add_argument("--stair-rise-min", type=float, default=STAIR_RISE_RANGE[0])
    parser.add_argument("--stair-rise-max", type=float, default=STAIR_RISE_RANGE[1])
    parser.add_argument("--stair-tread-min", type=float, default=STAIR_TREAD_RANGE[0])
    parser.add_argument("--stair-tread-max", type=float, default=STAIR_TREAD_RANGE[1])
    parser.add_argument("--stair-width-min", type=float, default=STAIR_WIDTH_RANGE[0])
    parser.add_argument("--stair-width-max", type=float, default=STAIR_WIDTH_RANGE[1])
    args = parser.parse_args()

    if args.stair_min_steps <= 0 or args.stair_min_steps > args.stair_max_steps:
        parser.error("--stair-min-steps must be positive and <= --stair-max-steps")
    for min_name, max_name in [
        ("stair_rise_min", "stair_rise_max"),
        ("stair_tread_min", "stair_tread_max"),
        ("stair_width_min", "stair_width_max"),
    ]:
        min_value = getattr(args, min_name)
        max_value = getattr(args, max_name)
        if min_value <= 0 or min_value > max_value:
            parser.error(f"--{min_name.replace('_', '-')} must be positive and <= --{max_name.replace('_', '-')}")

    types = list(GENERATORS.keys()) if args.type == "all" else [args.type]

    total = 0
    for terrain_type in types:
        gen_func, default_seed = GENERATORS[terrain_type]
        base_seed = args.seed if args.seed is not None else default_seed

        print(f"\nGenerating {args.num} {terrain_type} terrains (seed={base_seed})...")
        for i in range(args.num):
            seed = base_seed + i
            name = f"{terrain_type}_{i:03d}"
            out_dir = os.path.join(args.output_dir, name)

            kwargs = {}
            if terrain_type == "stairs":
                kwargs["uniform"] = args.uniform
                kwargs["step_count_range"] = (args.stair_min_steps, args.stair_max_steps)
                kwargs["rise_range"] = (args.stair_rise_min, args.stair_rise_max)
                kwargs["tread_range"] = (args.stair_tread_min, args.stair_tread_max)
                kwargs["width_range"] = (args.stair_width_min, args.stair_width_max)
            pieces = gen_func(seed, **kwargs)
            export_terrain(pieces, out_dir, name)
            total += 1

            if (i + 1) % 50 == 0 or i == args.num - 1:
                print(f"  {i + 1}/{args.num} done")

    print(f"\nDone. Generated {total} terrain assets to {args.output_dir}/")


if __name__ == "__main__":
    main()
