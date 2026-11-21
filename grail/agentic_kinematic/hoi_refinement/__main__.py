"""Native Stage 2 edit rounds; heavy model evaluation uses the configured runtime."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("run", "Edit/refresh a new unreviewed checkpoint"),
        ("repair", "Propose a bounded local repair, then independently review and preview"),
    ):
        run_parser = commands.add_parser(command, help=help_text)
        run_parser.add_argument("--config", required=True)
        run_parser.add_argument("--output", required=True)
        run_parser.add_argument("--input-blend")
        if command == "run":
            run_parser.add_argument("--operations")
    args = parser.parse_args(argv)
    from .round import run

    print(
        json.dumps(
            run(
                args.config,
                args.output,
                input_blend=args.input_blend,
                operations_path=getattr(args, "operations", None),
                repair=args.command == "repair",
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
