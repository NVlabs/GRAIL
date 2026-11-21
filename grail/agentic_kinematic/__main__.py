"""Run with ``python -m grail.agentic_kinematic --help``."""

import argparse
import json

from . import manifest


def main():
    parser = argparse.ArgumentParser(
        description="GRAIL evidence-guided kinematic candidates and review"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Bind the fixed benchmark and its original fallbacks")
    init.add_argument("--spec", required=True)
    init.add_argument("--output", required=True)
    check = sub.add_parser(
        "verify", help="Verify all file hashes, denominator and selected pairing"
    )
    check.add_argument("manifest")
    for name in ("register", "select"):
        cmd = sub.add_parser(name)
        cmd.add_argument("manifest")
        cmd.add_argument("--case", required=True)
        cmd.add_argument(
            "--output", required=True, help="New manifest; never overwrite a checkpoint"
        )
        if name == "register":
            cmd.add_argument("--record", required=True)
        else:
            cmd.add_argument("--candidate", required=True)
            cmd.add_argument("--review", required=True)
    page = sub.add_parser(
        "page", help="Build a portable English page with all cases and playable videos"
    )
    page.add_argument("manifest")
    page.add_argument("--output", required=True)
    cmd = sub.add_parser("fit-robot", help="Run bounded robot refinement")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "init":
        result = manifest.create(args.spec)
        manifest.write_json(args.output, result)
        result = {"manifest": args.output, "cases": result["expected_cases"]}
    elif args.command == "verify":
        result = manifest.verify(manifest.read_json(args.manifest))
    elif args.command in ("register", "select"):
        current = manifest.read_json(args.manifest)
        if args.command == "register":
            result = manifest.register(current, args.case, args.record)
        else:
            result = manifest.select(current, args.case, args.candidate, args.review)
        manifest.write_json(args.output, result)
        result = {"manifest": args.output}
    elif args.command == "page":
        from .media import build_page

        result = build_page(manifest.read_json(args.manifest), args.output)
    else:
        from .robot import run

        result = run(args.config, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
