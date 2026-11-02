"""End-to-end SMPL-X -> Unitree G1 retargeting pipeline.

Four stages:

  1. **retarget** — IK-solve SMPL-X poses onto the G1 skeleton via GMR
     (`grail.retargeting.retarget`). Output:
     ``data/motion_lib/<output>/{robot,objects,object_usd,meta}/``.
  2. **process** — derive hand actions + table geometry, write a
     training-ready `_ha` variant
     (`grail.retargeting.process`). Output:
     ``data/motion_lib/<output>_ha/{robot,objects,meta}/``.
  3. **compute_bps** *(skipped for single-object datasets)* — BPS shape
     encoding for object meshes (`grail.retargeting.compute_bps`).
     Output: ``data/motion_lib/<output>/bps/``.
  4. **verify** — sanity-check that ``robot/objects/meta/object_usd`` all carry
     the same set of stems; if ``bps/`` exists, print BPS counts and match stems to
     ``object_usd/``. Same for ``<output>_ha/`` when present (even if process or
     BPS was skipped this run). Hard-fails on mismatch — SONIC training requires
     identical stems under ``use_paired_motions: true``.

This module mirrors ``grail/retargeting/scripts/retarget_pipeline.sh`` but
runs entirely in-process via subprocess calls to the per-step modules so
the same CLI surface works with external schedulers without bash.

Usage (from repo root):

    python -m grail.pipelines.retarget \\
        --data_dir <recon_dir> \\
        --output_folder <name> \\
        [retarget passthrough flags ...]

Skip individual stages with ``--no_process`` / ``--no_bps`` / ``--no_verify``
(e.g. when iterating on a retarget-only step). Full docs:
``docs/source/retargeting.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_META_PKL = _REPO_ROOT / "data/body_profiles/unitree_g1/smplx/g1_skeleton_meta.pkl"


def _run(label: str, cmd: list[str]) -> None:
    """Run a subprocess; raise on non-zero exit."""
    print(f">>> {label}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _is_single_hoi_folder(data_dir: str) -> bool:
    """True when data_dir points at a single HOI sequence (has hoi_data/hoi_data.pkl)."""
    return (Path(data_dir) / "hoi_data" / "hoi_data.pkl").is_file()


def _step_retarget(data_dir: str, output_base: str, extra_args: list[str]) -> None:
    if _is_single_hoi_folder(data_dir):
        cmd = [
            sys.executable,
            "-m",
            "grail.retargeting.retarget",
            "--file",
            data_dir,
            "--robot",
            "unitree_g1",
            "--output_dir",
            output_base,
            "--no_viewer",
            *extra_args,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "grail.retargeting.retarget",
            "--data_dir",
            data_dir,
            "--all",
            "--robot",
            "unitree_g1",
            "--output_dir",
            output_base,
            "--no_viewer",
            *extra_args,
        ]
    _run("[1/4] retarget", cmd)


def _step_process(
    input_dir: str,
    output_dir: str,
    treat_hands_equally: bool = False,
    grasp_anticipation_frames: int = 10,
) -> None:
    # Mirrors grail/retargeting/scripts/process.sh.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grail.retargeting.process",
        "--input",
        input_dir,
        "--output",
        output_dir,
        "--meta_pkl",
        str(_META_PKL),
        "--include_contact_points",
        "--grasp_from_lift",
        "--lift_threshold",
        "0.02",
        "--grasp_anticipation_frames",
        str(grasp_anticipation_frames),
        "--skip_no_lift",
        "--per_object",
    ]
    if treat_hands_equally:
        cmd.append("--treat_hands_equally")
    _run("[2/4] process (hand actions + table geometry)", cmd)


def _link_object_usd(output_base: str, output_ha: str) -> None:
    """Create <output_ha>/object_usd -> ../<output_base_name>/object_usd symlink (idempotent)."""
    ha_path = Path(output_ha)
    if not ha_path.is_dir():
        print(f"    skipped object_usd symlink: {output_ha} does not exist")
        return
    link_path = ha_path / "object_usd"
    target = f"../{Path(output_base).name}/object_usd"
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            print(f"    skipped object_usd symlink: {link_path} exists and is a real directory")
            return
    link_path.symlink_to(target)
    print(f"    symlinked {link_path} -> {target}")


def _step_compute_bps(output_base: str) -> None:
    # Mirrors grail/retargeting/scripts/compute_bps.sh.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grail.retargeting.compute_bps",
        "--object_usd_dir",
        os.path.join(output_base, "object_usd"),
        "--output_dir",
        os.path.join(output_base, "bps"),
    ]
    _run("[3/4] compute_bps", cmd)


def _stems(d: Path, ext: str, exclude: set[str] = frozenset()) -> set[str]:
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob(f"*{ext}") if p.is_file() and p.stem not in exclude}


def _step_verify(output_base: str, output_ha: str) -> None:
    """Cross-check stems across the per-motion subdirs of the output library.

    SONIC's ``use_paired_motions: true`` requires N_robot == N_objects ==
    N_meta == N_object_usd with identical stems in identical order. This
    is the cheapest place to catch a missing or extra file before training.
    """
    print(">>> [4/4] verify (counts + stems)")
    base = Path(output_base)
    ha = Path(output_ha)

    robot = _stems(base / "robot", ".pkl")
    objects = _stems(base / "objects", ".pkl")
    meta = _stems(base / "meta", ".pkl")
    usd = _stems(base / "object_usd", ".usd")
    print(
        f"    {output_base}: robot={len(robot)} objects={len(objects)} "
        f"meta={len(meta)} object_usd={len(usd)}"
    )

    failures: list[str] = []
    expected = robot
    for name, got in [("objects", objects), ("meta", meta), ("object_usd", usd)]:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing or extra:
            preview_missing = missing[:5] + (["..."] if len(missing) > 5 else [])
            preview_extra = extra[:5] + (["..."] if len(extra) > 5 else [])
            failures.append(
                f"  {name}/ vs robot/: missing {len(missing)} {preview_missing}; "
                f"extra {len(extra)} {preview_extra}"
            )

    if (base / "bps").is_dir():
        # compute_bps writes <stem>.npy per object plus a sentinel _basis.npy.
        bps = _stems(base / "bps", ".npy", exclude={"_basis"})
        print(f"    {output_base}/bps: {len(bps)}")
        missing = sorted(usd - bps)
        extra = sorted(bps - usd)
        if missing or extra:
            failures.append(
                f"  bps/ vs object_usd/: missing {len(missing)} {missing[:5]}; "
                f"extra {len(extra)} {extra[:5]}"
            )

    if ha.is_dir():
        ha_robot = _stems(ha / "robot", ".pkl")
        ha_objects = _stems(ha / "objects", ".pkl")
        ha_meta = _stems(ha / "meta", ".pkl")
        print(
            f"    {output_ha}: robot={len(ha_robot)} objects={len(ha_objects)} "
            f"meta={len(ha_meta)}"
        )
        for name, got in [("objects", ha_objects), ("meta", ha_meta)]:
            missing = sorted(ha_robot - got)
            extra = sorted(got - ha_robot)
            if missing or extra:
                failures.append(
                    f"  _ha/{name}/ vs _ha/robot/: missing {len(missing)} {missing[:5]}; "
                    f"extra {len(extra)} {extra[:5]}"
                )
    else:
        print(f"    {output_ha}: (directory not present)")

    if failures:
        print("    FAIL: stem mismatches detected:")
        for f in failures:
            print(f)
        sys.exit(1)
    print("    OK: counts and stems consistent")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Recon-output directory (e.g. data/genhoi/<dataset>/generation/4dhoi_recon_valid/Hunyuan)",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Short name; outputs land at data/motion_lib/<output_folder>{,_ha}/",
    )
    parser.add_argument(
        "--no_retarget",
        action="store_true",
        help="Skip stage 1 (retarget IK) — use to run only stages 2/3 after a chunked run",
    )
    parser.add_argument(
        "--no_process", action="store_true", help="Skip stage 2 (hand actions / table)"
    )
    parser.add_argument(
        "--no_bps",
        action="store_true",
        help="Skip stage 3 (BPS encoding) — auto-skipped for single-object datasets",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip stage 4 (cross-check counts + stems across robot/objects/meta/object_usd/bps).",
    )
    parser.add_argument(
        "--treat_hands_equally",
        action="store_true",
        help=(
            "Pass through to process.py: preserve both arms and derive left/right hand "
            "actions symmetrically from their own contacts."
        ),
    )
    parser.add_argument(
        "--num_job_chunks",
        type=int,
        default=1,
        help="Total number of parallel chunks (forwarded to retarget stage).",
    )
    parser.add_argument(
        "--job_chunk_idx",
        type=int,
        default=0,
        help="Zero-based index of this chunk (0 .. num_job_chunks-1).",
    )
    parser.add_argument(
        "--grasp_anticipation_frames",
        type=int,
        default=10,
        help="Start right-hand closing this many frames before derived contact/lift.",
    )
    args, retarget_extra = parser.parse_known_args()

    output_base = f"data/motion_lib/{args.output_folder}"
    output_ha = f"{output_base}_ha"
    is_chunked = args.num_job_chunks > 1

    # Stage 1
    if args.no_retarget:
        print(">>> [1/4] retarget: skipped (--no_retarget)")
    else:
        chunk_args: list[str] = []
        if is_chunked:
            chunk_args = [
                "--num_job_chunks",
                str(args.num_job_chunks),
                "--job_chunk_idx",
                str(args.job_chunk_idx),
            ]
        _step_retarget(args.data_dir, output_base, retarget_extra + chunk_args)

    # Stages 2-4 operate on the full output directory and should only run
    # once after all chunks have finished, not inside each chunk.
    if is_chunked and not args.no_retarget:
        print(
            f">>> [2/4] process: skipped (chunked run {args.job_chunk_idx}/{args.num_job_chunks})"
        )
        print(">>> [3/4] compute_bps: skipped (chunked run)")
        print(">>> [4/4] verify: skipped (chunked run)")
        print("    After all chunks finish, run stages 2-4 with:")
        print(
            f"    python -m grail.pipelines.retarget --data_dir {args.data_dir}"
            f" --output_folder {args.output_folder} --no_retarget"
        )
        return

    # Stage 2
    if args.no_process:
        print(">>> [2/4] process: skipped (--no_process)")
    else:
        _step_process(
            output_base,
            output_ha,
            treat_hands_equally=args.treat_hands_equally,
            grasp_anticipation_frames=args.grasp_anticipation_frames,
        )
        _link_object_usd(output_base, output_ha)

    # Stage 3 (conditional)
    if args.no_bps:
        print(">>> [3/4] compute_bps: skipped (--no_bps)")
    else:
        usd_dir = Path(output_base) / "object_usd"
        usd_count = (
            len([p for p in usd_dir.glob("*.usd") if p.is_file()]) if usd_dir.is_dir() else 0
        )
        if usd_count > 1:
            _step_compute_bps(output_base)
        else:
            print(f">>> [3/4] compute_bps: skipped (single-object dataset, {usd_count} USDs)")

    # Stage 4
    if args.no_verify:
        print(">>> [4/4] verify: skipped (--no_verify)")
    else:
        _step_verify(output_base, output_ha)

    print(f"Pipeline complete. See data/motion_lib/{args.output_folder}{{,_ha}}/")


if __name__ == "__main__":
    main()
