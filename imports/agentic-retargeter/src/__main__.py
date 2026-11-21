"""Run with ``python -m agentic_retargeter --help``."""

import argparse
import json

from . import manifest


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Retargeter: native robot candidates and evidence-bound review"
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
    fit = sub.add_parser(
        "fit-robot", help="Run experimental bounded robot refinement; exports an UNREVIEWED candidate"
    )
    fit.add_argument("--config", required=True)
    fit.add_argument("--output", required=True)
    collision = sub.add_parser("collision", help="Query explicit mesh pairs through external pinned Newton; diagnostic only")
    collision.add_argument("--input", required=True, help="Bound original-mesh snapshot JSON")
    collision.add_argument("--runtime", required=True, help="External Python/repository/device JSON")
    collision.add_argument("--output", required=True, help="New query directory")
    repair = sub.add_parser("local-repair", help="Inspect geometry coverage, audit or repair a native motion window")
    repair.add_argument("action", choices=("coverage", "audit", "repair"))
    repair.add_argument("--config", required=True)
    repair.add_argument("--output", required=True, help="New output directory")
    from .blender_edit import add_cli

    add_cli(sub)
    from .video_cli import add_cli as add_video_cli

    add_video_cli(sub)
    from .retarget import add_cli as add_retarget_cli

    add_retarget_cli(sub)
    from .retarget_native_cli import add_cli as add_native_cli

    add_native_cli(sub)
    from .graspgen import add_cli as add_graspgen_cli

    add_graspgen_cli(sub)
    args = parser.parse_args()
    if args.command == "local-repair":
        from .local_repair import main as repair_main

        return repair_main([args.action, "--config", args.config, "--output", args.output])
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
    elif args.command == "collision":
        from pathlib import Path
        from .collision import run_query

        try:
            result = run_query(json.loads(Path(args.input).read_text()),
                               json.loads(Path(args.runtime).read_text()), args.output)
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(2, f"Collision query error: {error}\n")
    elif args.command == "blender-edit":
        from .blender_edit import BlenderEditError, run_cli

        try:
            result = run_cli(args)
        except (BlenderEditError, ValueError, OSError) as error:
            parser.exit(2, f"Blender edit error: {error}\n")
    elif args.command == "video":
        from .video_cli import run_cli

        try:
            result = run_cli(args)
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(2, f"Video analysis error: {error}\n")
    elif args.command == "retarget-native":
        from .retarget_native_cli import run_cli

        try:
            result = run_cli(args)
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            parser.exit(2, f"Native handoff error: {error}\n")
    elif args.command == "graspgenx":
        from .graspgen import run_cli

        try:
            result = run_cli(args)
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            parser.exit(2, f"GraspGenX error: {error}\n")
    elif args.command == "retarget":
        from .retarget import run_cli

        try:
            result = run_cli(args)
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            parser.exit(2, f"Retarget error: {error}\n")
    else:
        from .robot import run

        result = run(args.config, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command in ("retarget", "graspgenx") and args.action == "preflight" and not result["ok"]:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
