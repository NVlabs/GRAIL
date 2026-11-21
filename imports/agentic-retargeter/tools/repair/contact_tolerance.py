"""Reclassify saved contact measurements without rerunning FK/collision tests."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.contact_tolerance import evaluate_contact_tolerance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSON with rows, fps, required_frames, contacts, policies and provenance")
    p.add_argument("--output", required=True, help="New JSON report path")
    args = p.parse_args()
    raw = Path(args.input).read_bytes()
    data = json.loads(raw)
    if not data["policies"] or set(data["policies"]) - {"gap", "normal"}:
        raise ValueError("policies must contain gap and/or normal")
    reviews = {metric:evaluate_contact_tolerance(data["rows"], policy=policy, metric=metric, fps=data["fps"],
                                               required_frames=data["required_frames"], contacts=data["contacts"])
               for metric,policy in data["policies"].items()}
    report = dict(reviews=reviews, passed=all(r["passed"] for r in reviews.values()),
                  complete=all(r["complete"] for r in reviews.values()))
    report.update(scope="Configured contact metrics only; no motion acceptance or collision reclassification",
                  input_sha256=hashlib.sha256(raw).hexdigest(), provenance=data.get("provenance", {}),
                  code_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/"src/contact_tolerance.py").read_bytes()).hexdigest())
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"contact_metrics_passed": report["passed"], "complete": report["complete"], "output": args.output}))


if __name__ == "__main__":
    main()
