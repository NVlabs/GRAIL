#!/usr/bin/env python3
"""Merge shard export outputs into a single directory.

Usage:
    python merge_exports.py --exp_dir /path/to/experiment --source_data /path/to/source --step 50000
    python merge_exports.py --exp_dir /path/to/experiment --source_data /path/to/source --step 50000 --num_shards 8
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def count_files(directory: Path, pattern: str) -> int:
    """Count files matching a glob pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def copy_no_clobber(src: Path, dst: Path) -> int:
    """Recursively copy files from src to dst without overwriting. Walks subdirectories
    so nested assets (e.g. object_usd/textures/<motion>/model.jpg for chair USDs)
    propagate through the merge step. Returns total count of files copied."""
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(src.iterdir()):
        if f.is_file():
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
                copied += 1
        elif f.is_dir():
            copied += copy_no_clobber(f, dst / f.name)
    return copied


def merge_manifests(shard_manifests: list[dict], num_shards: int) -> dict:
    """Merge per-shard export_manifest.json files into a single manifest."""
    merged = {
        "num_shards": num_shards,
        "total_motions_evaluated": 0,
        "exported_count": 0,
        "skipped_no_trajectory": 0,
        "success_rate": 0.0,
        "exported_motions": [],
    }

    for manifest in shard_manifests:
        merged["total_motions_evaluated"] += manifest.get("total_motions_evaluated", 0)
        merged["exported_count"] += manifest.get("exported_count", 0)
        merged["skipped_no_trajectory"] += manifest.get("skipped_no_trajectory", 0)
        merged["exported_motions"].extend(manifest.get("exported_motions", []))

    if merged["total_motions_evaluated"] > 0:
        merged["success_rate"] = merged["exported_count"] / merged["total_motions_evaluated"]

    return merged


def motion_keys(shard_dir: Path) -> set[str]:
    """Motion-key set for a shard = stems of robot/*.pkl files."""
    robot_dir = shard_dir / "robot"
    if not robot_dir.exists():
        return set()
    return {f.stem for f in robot_dir.glob("*.pkl")}


def main():
    parser = argparse.ArgumentParser(
        description="Merge shard export outputs into a single directory"
    )
    parser.add_argument("--exp_dir", required=True, help="Absolute path to experiment directory")
    parser.add_argument("--source_data", required=True, help="Path to source data directory")
    parser.add_argument("--step", required=True, help="Training step (e.g. 50000)")
    parser.add_argument("--num_shards", type=int, default=5, help="Number of shards (default: 5)")
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.5,
        help="Pairwise motion-key overlap (vs min shard size) above which merge FAILs (default: 0.5)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    source_data = Path(args.source_data)
    step = args.step
    num_shards = args.num_shards

    base_dir = exp_dir / "exported" / f"step_{step}"
    merged_dir = base_dir / "merged"

    print(f"Experiment dir : {exp_dir}")
    print(f"Source data    : {source_data}")
    print(f"Step           : {step}")
    print(f"Num shards     : {num_shards}")
    print(f"Base dir       : {base_dir}")
    print(f"Merged dir     : {merged_dir}")
    print()

    # Create merged subdirectories
    for subdir in ["robot", "objects", "vis", "object_usd", "meta"]:
        (merged_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Collect per-shard stats and manifests
    shard_stats = []
    shard_manifests = []
    missing_shards = []

    for rank in range(num_shards):
        shard_dir = base_dir / f"shard_{rank}"
        if not shard_dir.exists():
            missing_shards.append(rank)
            shard_stats.append(
                {
                    "rank": rank,
                    "exists": False,
                    "robot": 0,
                    "objects": 0,
                    "vis": 0,
                    "motion_keys": set(),
                }
            )
            continue

        robot_count = count_files(shard_dir / "robot", "*.pkl")
        objects_count = count_files(shard_dir / "objects", "*.pkl")
        vis_count = count_files(shard_dir / "vis", "*.mp4")

        shard_stats.append(
            {
                "rank": rank,
                "exists": True,
                "robot": robot_count,
                "objects": objects_count,
                "vis": vis_count,
                "motion_keys": motion_keys(shard_dir),
            }
        )

        # Copy files (no-clobber)
        copy_no_clobber(shard_dir / "robot", merged_dir / "robot")
        copy_no_clobber(shard_dir / "objects", merged_dir / "objects")
        copy_no_clobber(shard_dir / "vis", merged_dir / "vis")
        copy_no_clobber(shard_dir / "object_usd", merged_dir / "object_usd")
        copy_no_clobber(shard_dir / "meta", merged_dir / "meta")

        # Load manifest if present
        manifest_path = shard_dir / "export_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                shard_manifests.append(json.load(f))

    # Copy auxiliary files from source_data to dataset ROOT (not object_usd/) so
    # object_usd/*.usd count matches robot/objects counts 1:1. flat_placeholder.usd is
    # required by terrain env pairing; training code checks dataset root first, then
    # falls back to object_usd/ for backward compatibility.
    for aux_name in ["config.yaml", "flat_placeholder.usd"]:
        src_aux = source_data / "object_usd" / aux_name
        if src_aux.exists():
            dst_aux = merged_dir / aux_name
            if not dst_aux.exists():
                shutil.copy2(src_aux, dst_aux)
                print(f"Copied {aux_name} to dataset root from source_data")
        else:
            print(f"No object_usd/{aux_name} found in source_data (skipped)")
    print()

    # Print per-shard summary table
    print("=" * 70)
    print(f"{'Shard':>8} {'Status':>10} {'Robot':>8} {'Objects':>8} {'Vis':>8}")
    print("-" * 70)
    for s in shard_stats:
        status = "OK" if s["exists"] else "MISSING"
        print(f"{s['rank']:>8} {status:>10} {s['robot']:>8} {s['objects']:>8} {s['vis']:>8}")
    print("=" * 70)
    print()

    if missing_shards:
        print(f"WARNING: Missing shards: {missing_shards}")
        print()

    # Count final merged files
    merged_robot = count_files(merged_dir / "robot", "*.pkl")
    merged_objects = count_files(merged_dir / "objects", "*.pkl")
    merged_vis = count_files(merged_dir / "vis", "*.mp4")
    merged_usd = count_files(merged_dir / "object_usd", "*.usd")
    merged_meta = count_files(merged_dir / "meta", "*.pkl")

    print("Merged totals:")
    print(f"  robot/*.pkl     : {merged_robot}")
    print(f"  objects/*.pkl   : {merged_objects}")
    print(f"  vis/*.mp4       : {merged_vis}")
    print(f"  object_usd/*.usd: {merged_usd}")
    print(f"  meta/*.pkl      : {merged_meta}")
    print()

    # Verify counts match (per-motion USDs must match robot/objects 1:1)
    merge_counts_pass = merged_robot == merged_objects == merged_vis == merged_usd
    if merge_counts_pass:
        print(f"PASS: robot == objects == vis == object_usd/*.usd == {merged_robot}")
    else:
        print(
            f"FAIL: count mismatch - robot={merged_robot}, objects={merged_objects}, "
            f"vis={merged_vis}, object_usd/*.usd={merged_usd}"
        )

    # ---- Cross-shard partition sanity checks ----
    # Catches stale Phase 1 shards (e.g. num_shards changed between re-submissions)
    # that silently replicate partitions into the merged output.
    print()
    print("Partition sanity checks:")
    shard_key_sets = [(s["rank"], s["motion_keys"]) for s in shard_stats if s["exists"]]
    unique_keys: set[str] = set()
    for _, ks in shard_key_sets:
        unique_keys |= ks
    sum_shard_exported = sum(len(ks) for _, ks in shard_key_sets)
    print(f"  unique_motions_exported : {len(unique_keys)}")
    print(f"  sum_shard_exported      : {sum_shard_exported}")
    if sum_shard_exported != len(unique_keys):
        dup_count = sum_shard_exported - len(unique_keys)
        print(
            f"  WARN: {dup_count} motion-name duplicates across shards — "
            f"no-clobber dedup masked overlap. Check for stale phase1_shard_* dirs."
        )

    # Pairwise overlap — flags partitions that cover the same motion subset.
    overlap_fails = []
    for i in range(len(shard_key_sets)):
        rank_i, keys_i = shard_key_sets[i]
        for j in range(i + 1, len(shard_key_sets)):
            rank_j, keys_j = shard_key_sets[j]
            inter = keys_i & keys_j
            if not inter:
                continue
            denom = min(len(keys_i), len(keys_j))
            if denom == 0:
                continue
            frac = len(inter) / denom
            tag = "FAIL" if frac >= args.overlap_threshold else "INFO"
            msg = (
                f"  {tag}: shard {rank_i} ({len(keys_i)}) vs shard {rank_j} ({len(keys_j)}): "
                f"{len(inter)} shared motions ({frac:.1%} of smaller shard)"
            )
            print(msg)
            if tag == "FAIL":
                overlap_fails.append((rank_i, rank_j, frac))
    if overlap_fails:
        print(
            f"  FAIL: {len(overlap_fails)} shard pair(s) have >= {args.overlap_threshold:.0%} overlap — "
            f"likely stale partitions from a prior Phase 1 run with different --num_shards. "
            f"Re-run submit_phase1.py with --clean."
        )

    # Coverage vs source library
    source_robot = source_data / "robot"
    if source_robot.exists():
        source_count = count_files(source_robot, "*.pkl")
        if source_count > 0:
            coverage = len(unique_keys) / source_count
            print(
                f"  source_library_size     : {source_count}  "
                f"(coverage = {len(unique_keys)}/{source_count} = {coverage:.1%} = true SR)"
            )
    else:
        print(f"  source_library_size     : (source_data/robot not found at {source_robot})")

    # Merge manifests and write
    if shard_manifests:
        merged_manifest = merge_manifests(shard_manifests, num_shards)
        manifest_out = merged_dir / "export_manifest.json"
        with open(manifest_out, "w") as f:
            json.dump(merged_manifest, f, indent=2)

        print()
        print("Merged manifest:")
        print(f"  total_motions_evaluated : {merged_manifest['total_motions_evaluated']}")
        print(f"  exported_count          : {merged_manifest['exported_count']}")
        print(f"  skipped_no_trajectory   : {merged_manifest['skipped_no_trajectory']}")
        print(f"  success_rate            : {merged_manifest['success_rate']:.4f}")
        print(f"  exported_motions        : {len(merged_manifest['exported_motions'])} entries")
        print(f"  Written to: {manifest_out}")
    else:
        print()
        print("WARNING: No export_manifest.json found in any shard, skipping manifest merge")

    print()
    print(f"Done. Merged output at: {merged_dir}")

if __name__ == "__main__":
    main()
