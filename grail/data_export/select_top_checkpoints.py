"""Select top-K training checkpoints from a W&B run/group by eval success rate.

Ranks checkpoints by `eval/success/success_rate` (descending). Tiebreak metric
defaults to `eval/all/mpjpe_l` (ascending); pass `--hoi` to use
`eval/all/obj_pos_error` (ascending) instead — mpjpe_l is not meaningful for
HOI experiments because the policy optimizes object pose, not body pose.

Each eval row carries an `eval_step` value that maps 1:1 to a
`model_step_{eval_step}.pt` checkpoint filename.

For terrain/sitting sweeps (tnt / tns / tnc / tnch / tnf) the training curriculum
filters to terrain-relevant motions, so the global `eval/success/*` metric is
already the terrain-relevant metric — no per-category slice needed.

Usage:
    # Single run
    python -m grail.data_export.select_top_checkpoints \\
        --run ENTITY/PROJECT/RUN_ID --k 5

    # W&B group (picks the finished run with the most eval rows)
    python -m grail.data_export.select_top_checkpoints \\
        --entity ENTITY --project PROJECT --group GROUP_NAME --k 5

    # Write JSON output
    python -m grail.data_export.select_top_checkpoints \\
        --entity ENTITY --project PROJECT --group GROUP_NAME --k 5 \\
        --out out/top_checkpoints.json
"""

import argparse
import json
import sys
from pathlib import Path


SR_KEY = "eval/success/success_rate"
MPJPE_L_KEY = "eval/all/mpjpe_l"
MPJPE_G_KEY = "eval/all/mpjpe_g"
OBJ_POS_KEY = "eval/all/obj_pos_error"
STEP_KEY = "eval_step"


def parse_args():
    p = argparse.ArgumentParser(
        description="Select top-K checkpoints from W&B by eval success rate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="W&B run path, e.g. ENTITY/PROJECT/RUN_ID")
    src.add_argument("--group", help="W&B group name; picks the best finished run in the group")
    p.add_argument("--entity", help="Account or team for a group or short run ID")
    p.add_argument("--project", help="Project for a group or short run ID")
    p.add_argument("--k", type=int, default=5, help="Top-K checkpoints (default: 5)")
    p.add_argument("--hoi", action="store_true",
                   help="HOI experiment: tiebreak on obj_pos_error (asc) instead of mpjpe_l")
    p.add_argument(
        "--min_sr", type=float, default=0.0,
        help="Drop rows below this success rate (default: 0.0)",
    )
    p.add_argument(
        "--step_pad", type=int, default=6,
        help="Zero-pad eval_step to this width in checkpoint filename "
             "(default: 6 = zero-pad to 6 digits, e.g. model_step_013500.pt; set to 0 for unpadded)",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Write JSON payload to this path (also prints a summary to stdout)",
    )
    p.add_argument(
        "--samples", type=int, default=5000,
        help="Max history rows to fetch per run (default: 5000)",
    )
    args = p.parse_args()
    if args.run and "/" in args.run:
        if len(args.run.split("/")) != 3 or not all(args.run.split("/")):
            p.error("--run expects ENTITY/PROJECT/RUN_ID or a short run ID")
    elif not (args.entity and args.project):
        p.error("--entity and --project are required for a group or short run ID")
    return args


def _import_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        print(
            "ERROR: wandb not installed. Activate your GRAIL/SONIC conda env\n"
            "  (pip install wandb) and rerun:\n"
            "  python -m grail.data_export.select_top_checkpoints ...",
            file=sys.stderr,
        )
        sys.exit(2)


def pick_group_run(api, entity, project, group, samples):
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}))
    if not runs:
        print(f"ERROR: no runs found in group '{group}' under {entity}/{project}",
              file=sys.stderr)
        sys.exit(1)

    scored = []
    for r in runs:
        try:
            hist = r.history(
                samples=samples, pandas=False,
                keys=[SR_KEY, MPJPE_L_KEY, OBJ_POS_KEY, STEP_KEY],
            )
        except Exception as e:
            print(f"  [skip] {r.id}: history error {e}", file=sys.stderr)
            continue
        n_eval = sum(1 for row in hist if row.get(SR_KEY) is not None)
        scored.append((r, n_eval))

    scored.sort(key=lambda x: (-x[1], x[0].state != "finished"))
    best, n_eval = scored[0]
    print(
        f"Group '{group}': {len(runs)} runs — picking {best.id} "
        f"(state={best.state}, eval rows={n_eval})",
        file=sys.stderr,
    )
    return best


def fetch_eval_rows(run, samples):
    hist = run.history(
        samples=samples, pandas=False,
        keys=[SR_KEY, MPJPE_L_KEY, MPJPE_G_KEY, OBJ_POS_KEY, STEP_KEY],
    )
    rows = []
    for r in hist:
        sr = r.get(SR_KEY)
        step = r.get(STEP_KEY)
        if sr is None or step is None:
            continue
        rows.append({
            "eval_step": int(step),
            "success_rate": float(sr),
            "mpjpe_l": float(r.get(MPJPE_L_KEY) or float("inf")),
            "mpjpe_g": float(r.get(MPJPE_G_KEY) or float("inf")),
            "obj_pos_error": float(r.get(OBJ_POS_KEY) or float("inf")),
        })
    return rows


def rank_rows(rows, min_sr, tiebreak_key="mpjpe_l"):
    filt = [r for r in rows if r["success_rate"] >= min_sr]
    filt.sort(key=lambda r: (-r["success_rate"], r[tiebreak_key]))
    return filt


def ckpt_filename(step, pad):
    if pad > 0:
        return f"model_step_{step:0{pad}d}.pt"
    return f"model_step_{step}.pt"


def main():
    args = parse_args()
    wandb = _import_wandb()
    api = wandb.Api()

    if args.run:
        run_path = args.run if "/" in args.run else f"{args.entity}/{args.project}/{args.run}"
        run = api.run(run_path)
    else:
        run = pick_group_run(api, args.entity, args.project, args.group, args.samples)

    run_path = f"{run.entity}/{run.project}/{run.id}"
    exp_dir = run.config.get("experiment_dir") or run.config.get("experiment_save_dir")

    print(f"Run:        {run_path}", file=sys.stderr)
    print(f"Name:       {run.name}", file=sys.stderr)
    print(f"State:      {run.state}", file=sys.stderr)
    print(f"Group:      {run.group}", file=sys.stderr)
    print(f"Exp dir:    {exp_dir}", file=sys.stderr)

    rows = fetch_eval_rows(run, args.samples)
    if not rows:
        print("ERROR: no eval rows with success_rate found", file=sys.stderr)
        sys.exit(1)

    tiebreak_key = "obj_pos_error" if args.hoi else "mpjpe_l"
    ranked = rank_rows(rows, args.min_sr, tiebreak_key=tiebreak_key)
    if not ranked:
        print(f"ERROR: no rows passed min_sr={args.min_sr}", file=sys.stderr)
        sys.exit(1)

    # Deduplicate by eval_step (keep best — the sorted first occurrence).
    seen = set()
    unique = []
    for r in ranked:
        if r["eval_step"] in seen:
            continue
        seen.add(r["eval_step"])
        unique.append(r)

    top = unique[: args.k]

    print(
        f"\nTop-{len(top)} of {len(unique)} eval points "
        f"(min_sr={args.min_sr}):",
        file=sys.stderr,
    )
    tb_label = "obj_pos" if args.hoi else "mpjpe_l"
    header = f"{'rank':>4} {'eval_step':>10} {'SR':>6} {tb_label:>9} {'mpjpe_g':>9} checkpoint"
    print(header, file=sys.stderr)
    for i, r in enumerate(top):
        fn = ckpt_filename(r["eval_step"], args.step_pad)
        tb_val = r["obj_pos_error"] if args.hoi else r["mpjpe_l"]
        print(
            f"{i+1:>4} {r['eval_step']:>10} {r['success_rate']:>6.3f} "
            f"{tb_val:>9.4f} {r['mpjpe_g']:>9.2f} {fn}",
            file=sys.stderr,
        )

    payload = {
        "run": run_path,
        "run_name": run.name,
        "group": run.group,
        "experiment_dir": exp_dir,
        "k": args.k,
        "min_sr": args.min_sr,
        "step_pad": args.step_pad,
        "hoi": bool(args.hoi),
        "tiebreak_key": tiebreak_key,
        "top": [
            {
                "rank": i + 1,
                "eval_step": r["eval_step"],
                "checkpoint": ckpt_filename(r["eval_step"], args.step_pad),
                "success_rate": r["success_rate"],
                "mpjpe_l": r["mpjpe_l"],
                "mpjpe_g": r["mpjpe_g"],
                "obj_pos_error": r["obj_pos_error"],
            }
            for i, r in enumerate(top)
        ],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out_path}", file=sys.stderr)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
