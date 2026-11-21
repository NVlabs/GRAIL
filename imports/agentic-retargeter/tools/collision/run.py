"""Query an explicit original-mesh snapshot in an external Newton runtime."""
import argparse
import json
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.collision import run_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--python", required=True, help="Absolute external Python executable")
    parser.add_argument("--repo", required=True, help="Pinned ho_contact_detection checkout")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    report = run_query(json.loads(args.snapshot.read_text()), {"python": args.python, "repo": args.repo, "device": args.device, "timeout_s": args.timeout}, args.output_dir)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir.resolve() / "report.json"), "pairs": [{"a": pair["mesh_a"], "b": pair["mesh_b"], "contact_count": pair["contact_count"]} for pair in report["pairs"]]}, indent=2))


if __name__ == "__main__":
    main()
