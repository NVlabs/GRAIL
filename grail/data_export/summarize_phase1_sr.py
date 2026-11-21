"""Aggregate Phase 1 eval shard results into real full-data success rates.

For each checkpoint step under {exp_dir}/eval/step_{step}/phase1_shard_{N}/,
reads every shard's metrics_eval.json and computes:
  - total motions across shards
  - total non-terminated (success) motions
  - real_success_rate = success / total
  - mean eval/all/mpjpe_l and mpjpe_g over all motions
  - mean eval/all/obj_pos_error over all motions (for HOI experiments)

Unlike the W&B value (computed on a max_unique_motions=88 subset per eval tick),
this SR is over the FULL motion library — the ground truth "real" SR.

Usage:
    # Summarize all step_* dirs under an experiment
    python -m grail.data_export.summarize_phase1_sr \\
        --exp_dir runs/example

    # Limit to specific steps
    python -m grail.data_export.summarize_phase1_sr \\
        --exp_dir ... --steps 13500 12500 8500 9500 9000

    # Write JSON + compare against a top-K JSON from select_top_checkpoints.py
    python -m grail.data_export.summarize_phase1_sr \\
        --exp_dir ... --compare out/grail-data-export/top_checkpoints.json \\
        --out out/grail-data-export/top_checkpoints_real_sr.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate Phase 1 shard metrics into per-checkpoint real SR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp_dir", required=True,
                   help="Experiment dir containing eval/step_*/phase1_shard_*")
    p.add_argument("--steps", nargs="+", default=None,
                   help="Limit to these step numbers (otherwise discover all)")
    p.add_argument("--compare", default=None,
                   help="JSON from select_top_checkpoints.py — emit side-by-side "
                        "W&B-subset SR vs real full-data SR")
    p.add_argument("--out", default=None, help="Write JSON payload to this path")
    p.add_argument("--missing_shards_fatal", action="store_true",
                   help="Exit non-zero if any shard is missing metrics_eval.json "
                        "(default: warn and include partial result)")
    p.add_argument("--hoi", action="store_true",
                   help="HOI experiment: rank by (real_SR desc, obj_pos_error asc) "
                        "instead of (real_SR desc, mpjpe_l asc)")
    return p.parse_args()


def discover_steps(exp_dir: str):
    pattern = os.path.join(exp_dir, "eval", "step_*")
    steps = []
    for d in sorted(glob.glob(pattern)):
        name = os.path.basename(d)
        if name.startswith("step_"):
            steps.append(name[len("step_"):])
    return steps


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def aggregate_step(exp_dir: str, step: str, missing_fatal: bool):
    step_dir = os.path.join(exp_dir, "eval", f"step_{step}")
    shard_dirs = sorted(glob.glob(os.path.join(step_dir, "phase1_shard_*")))

    total_motions = 0
    total_success = 0
    per_shard = []
    mpjpe_l_vals = []
    mpjpe_g_vals = []
    obj_pos_vals = []
    wandb_sr_vals = []

    unique_keys = set()
    for sd in shard_dirs:
        rank = os.path.basename(sd).split("_")[-1]
        metrics_path = os.path.join(sd, "metrics_eval.json")
        if not os.path.exists(metrics_path):
            msg = f"  [shard {rank}] MISSING {metrics_path}"
            print(msg, file=sys.stderr)
            if missing_fatal:
                sys.exit(1)
            per_shard.append({"rank": int(rank), "status": "missing"})
            continue

        try:
            d = json.load(open(metrics_path))
        except Exception as e:
            print(f"  [shard {rank}] PARSE ERROR: {e}", file=sys.stderr)
            if missing_fatal:
                sys.exit(1)
            per_shard.append({"rank": int(rank), "status": "parse_error"})
            continue

        all_dict = d.get("eval/all_metrics_dict", {}) or {}
        terminated = all_dict.get("terminated") or []
        motion_keys = all_dict.get("motion_keys") or []

        n_motions = len(motion_keys) if motion_keys else len(terminated)
        n_success = n_motions - int(sum(bool(t) for t in terminated))
        total_motions += n_motions
        total_success += n_success
        # Track unique motion keys across shards (if available)
        if motion_keys:
            for k in motion_keys:
                unique_keys.add(str(k))

        wandb_sr = d.get("eval/success/success_rate")
        if wandb_sr is not None:
            wandb_sr_vals.append(float(wandb_sr))
        mpjpe_l = d.get("eval/all/mpjpe_l")
        if mpjpe_l is not None:
            mpjpe_l_vals.append(float(mpjpe_l))
        mpjpe_g = d.get("eval/all/mpjpe_g")
        if mpjpe_g is not None:
            mpjpe_g_vals.append(float(mpjpe_g))
        obj_pos = d.get("eval/all/obj_pos_error")
        if obj_pos is not None:
            obj_pos_vals.append(float(obj_pos))

        per_shard.append({
            "rank": int(rank),
            "status": "ok",
            "n_motions": n_motions,
            "n_success": n_success,
            "shard_sr": (n_success / n_motions) if n_motions else None,
            "wandb_sr": float(wandb_sr) if wandb_sr is not None else None,
        })

    real_sr = (total_success / total_motions) if total_motions else None
    return {
        "step": step,
        "num_shards_found": len(shard_dirs),
        "num_shards_ok": sum(1 for s in per_shard if s["status"] == "ok"),
        "total_motions": total_motions,  # env-episodes across shards
        "total_success": total_success,
        "unique_motions": len(unique_keys) if unique_keys else None,
        "real_success_rate": real_sr,
        "mpjpe_l_mean": _mean(mpjpe_l_vals),
        "mpjpe_g_mean": _mean(mpjpe_g_vals),
        "obj_pos_error_mean": _mean(obj_pos_vals),
        "per_shard_wandb_sr_mean": _mean(wandb_sr_vals),
        "shards": per_shard,
    }


def main():
    args = parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"ERROR: exp_dir does not exist: {args.exp_dir}", file=sys.stderr)
        sys.exit(1)

    steps = args.steps or discover_steps(args.exp_dir)
    if not steps:
        print(f"ERROR: no step_* subdirs under {args.exp_dir}/eval/", file=sys.stderr)
        sys.exit(1)

    print(f"Exp dir: {args.exp_dir}", file=sys.stderr)
    print(f"Steps:   {steps}", file=sys.stderr)

    results = []
    for step in steps:
        print(f"\n=== step {step} ===", file=sys.stderr)
        r = aggregate_step(args.exp_dir, step, args.missing_shards_fatal)
        n_ok = r["num_shards_ok"]
        n_found = r["num_shards_found"]
        tm = r["total_motions"]
        ts = r["total_success"]
        sr = r["real_success_rate"]
        uniq = r.get("unique_motions")
        uniq_str = f"  unique: {uniq}" if uniq is not None else ""
        print(
            f"  shards: {n_ok}/{n_found} ok  "
            f"motions: {tm}  success: {ts}  "
            f"real_SR: {sr}{uniq_str}",
            file=sys.stderr,
        )
        results.append(r)

    # Compare against top-K JSON if provided
    compare_rows = None
    if args.compare:
        top = json.load(open(args.compare))
        by_step = {str(int(r["step"])): r for r in results}
        compare_rows = []
        for entry in top["top"]:
            key = str(int(entry["eval_step"]))
            agg = by_step.get(key) or {}
            compare_rows.append({
                "rank": entry["rank"],
                "eval_step": entry["eval_step"],
                "checkpoint": entry["checkpoint"],
                "wandb_subset_sr": entry["success_rate"],
                "wandb_subset_mpjpe_l": entry["mpjpe_l"],
                "wandb_subset_obj_pos_error": entry.get("obj_pos_error"),
                "real_success_rate": agg.get("real_success_rate"),
                "real_total_motions": agg.get("total_motions"),
                "real_total_success": agg.get("total_success"),
                "real_mpjpe_l_mean": agg.get("mpjpe_l_mean"),
                "real_obj_pos_error_mean": agg.get("obj_pos_error_mean"),
            })

    # Pretty-print ranking by real SR (tiebreak: obj_pos_error if HOI else mpjpe_l)
    def _tb(r):
        if args.hoi:
            v = r.get("obj_pos_error_mean")
        else:
            v = r.get("mpjpe_l_mean")
        return v if v is not None else float("inf")
    sorted_results = sorted(
        results, key=lambda r: (-(r["real_success_rate"] or 0), _tb(r))
    )
    tb_label = "obj_pos" if args.hoi else "mpjpe_l"
    print("\n=== Ranking by real full-data success rate ===", file=sys.stderr)
    hdr = "{:>4}  {:>8}  {:>8}  {:>7}  {:>5}  {:>8}".format(
        "rank", "step", "real_SR", "motions", "succ", tb_label)
    print(hdr, file=sys.stderr)
    for i, r in enumerate(sorted_results):
        sr = r["real_success_rate"]
        sr_str = f"{sr:.4f}" if sr is not None else "   n/a"
        tb_val = r["obj_pos_error_mean"] if args.hoi else r["mpjpe_l_mean"]
        tb_str = (f"{tb_val:.4f}" if args.hoi else f"{tb_val:.2f}") if tb_val is not None else "   n/a"
        step_v = r["step"]
        tm = r["total_motions"]
        ts = r["total_success"]
        print(
            f"{i+1:>4}  {step_v:>8}  {sr_str:>8}  "
            f"{tm:>7}  {ts:>5}  {tb_str:>8}",
            file=sys.stderr,
        )

    if compare_rows:
        print("\n=== W&B subset SR vs real full-data SR ===", file=sys.stderr)
        hdr = "{:>4}  {:>8}  {:>9}  {:>8}  {:>7}  {:>14}  {:>12}".format(
            "rank", "step", "subset_SR", "real_SR", "motions", "subset_mpjpe_l", "real_mpjpe_l")
        print(hdr, file=sys.stderr)
        for r in compare_rows:
            real = r["real_success_rate"]
            real_s = f"{real:.4f}" if real is not None else "   n/a"
            real_m = r["real_mpjpe_l_mean"]
            real_m_s = f"{real_m:.2f}" if real_m is not None else "   n/a"
            tot = r["real_total_motions"] or 0
            rr = r["rank"]
            es = r["eval_step"]
            wsr = r["wandb_subset_sr"]
            wmp = r["wandb_subset_mpjpe_l"]
            print(
                f"{rr:>4}  {es:>8}  {wsr:>9.4f}  "
                f"{real_s:>8}  {tot:>7}  "
                f"{wmp:>14.2f}  {real_m_s:>12}",
                file=sys.stderr,
            )

    payload = {
        "exp_dir": args.exp_dir,
        "results": results,
        "ranked_by_real_sr": [
            {"step": r["step"], "real_success_rate": r["real_success_rate"]}
            for r in sorted_results
        ],
        "compare": compare_rows,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out_path}", file=sys.stderr)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
