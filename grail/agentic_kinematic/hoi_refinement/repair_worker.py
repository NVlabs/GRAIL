"""Run one bounded native Stage 2 proposal in its selected model environment."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .contracts import binding, load_npz, read_json, verify_binding, write_json
from .repair import repair_plan
from .round import _model_bindings


def run_request(request, *, evaluator=None):
    """Save proposal_plan.json/report.json; preserve a failure report on errors."""
    output = Path(request["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path, report_path = output / "proposal_plan.json", output / "report.json"
    if plan_path.exists() or report_path.exists():
        raise FileExistsError(f"Preserve existing repair output: {output}")
    try:
        required = {
            "base_motion",
            "seed_plan",
            "scene",
            "model",
            "repair",
            "evidence",
            "output_dir",
        }
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(f"Repair request requires exactly {sorted(required)}")
        if Path(request["base_motion"]).suffix != ".npz":
            raise ValueError("Repair worker base_motion must be a safe canonical NPZ")
        inputs = {
            name: binding(request[name]) for name in ("base_motion", "seed_plan", "scene", "model")
        }
        model = read_json(request["model"])
        model_assets = _model_bindings(model)
        proposal, report = repair_plan(
            load_npz(request["base_motion"]),
            read_json(request["seed_plan"]),
            load_npz(request["scene"]),
            model,
            request["repair"],
            request["evidence"],
            evaluator=evaluator,
        )
        for record in [*inputs.values(), *model_assets]:
            verify_binding(record)
        report["bindings"]["worker_inputs"] = inputs
        report["bindings"]["model_assets"] = model_assets
        write_json(plan_path, proposal)
        report["proposal_plan"] = binding(plan_path)
        write_json(report_path, report)
        return report
    except Exception as exc:
        write_json(
            report_path,
            dict(
                schema="grail.hoi_local_repair.v1",
                status="UNREVIEWED",
                review_status="UNREVIEWED",
                accepted=False,
                stop_reason="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                collision=dict(status="NOT_PERFORMED", optimized=False),
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    run_request(read_json(args.request))


if __name__ == "__main__":
    main()
