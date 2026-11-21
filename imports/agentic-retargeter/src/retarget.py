"""Explicit source-model routing for initial robot retargeting.

SMPL-X uses the existing external GRAIL/GMR adapter. SOMA uses NVIDIA's
soma-retargeter. Neither adapter is a fallback for the other skeleton.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


BACKENDS = {"smplx": "gmr", "soma": "soma"}


def _backend(source_model):
    if not isinstance(source_model, str) or source_model not in BACKENDS:
        raise ValueError("source_model must be explicitly 'smplx' or 'soma'")
    name = BACKENDS[source_model]
    return name, importlib.import_module(f".{name}", __package__)


def plan(source_model, config_path, output):
    """Bind the selected backend's actual input contract without converting it."""
    name, backend = _backend(source_model)
    backend_plan = backend.plan(config_path, output)
    return {"schema_version": 1, "source_model": source_model, "backend": name,
            "status": backend_plan["status"], "backend_plan": backend_plan,
            "next_stage": "Inspect the native handoff and bind the original robot model before Blender editing.",
            "motion_status": "UNREVIEWED", "selected": False}


def _resolve(proposal):
    if not isinstance(proposal, dict) or proposal.get("schema_version") != 1:
        raise ValueError("Expected a retarget plan with schema_version=1")
    name, backend = _backend(proposal.get("source_model"))
    if proposal.get("backend") != name or not isinstance(proposal.get("backend_plan"), dict):
        raise ValueError("Source-model/backend pairing changed; create a new plan")
    return backend


def preflight(proposal):
    """Check the chosen runtime; never try another backend after an error."""
    backend = _resolve(proposal)
    checks = backend.preflight(proposal["backend_plan"])
    blockers = proposal["backend_plan"].get("blockers", [])
    ready = checks.get("ok") is True and not blockers
    return {"schema_version": 1, "source_model": proposal["source_model"],
            "backend": proposal["backend"], "ok": ready,
            "status": "READY_UNREVIEWED" if ready else "BLOCKED",
            "blockers": blockers, "checks": checks, "selected": False}


def execute(proposal, *, timeout=None):
    """Run the selected adapter and record one common, unreviewed handoff."""
    backend = _resolve(proposal)
    result = backend.execute(proposal["backend_plan"], timeout=timeout)
    output = Path(proposal["backend_plan"]["output"])
    report = {"schema_version": 1, "source_model": proposal["source_model"],
              "backend": proposal["backend"], "status": result["status"],
              "motion_status": "UNREVIEWED", "selected": False,
              "backend_report": str(output / "run.json"), "result": result}
    destination = output / "retarget_result.json"
    with destination.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"source_model": proposal["source_model"], "backend": proposal["backend"],
            "status": result["status"], "report": str(destination),
            "motion_status": "UNREVIEWED", "selected": False}


def add_cli(subparsers):
    parser = subparsers.add_parser("retarget", help="SMPL-X → GMR or SOMA → NVIDIA soma-retargeter")
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--source-model", required=True, choices=tuple(BACKENDS))
    parser.add_argument("--config", required=True, help="Selected backend's JSON configuration")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--timeout", type=float, default=None, help="Optional per-conversion timeout in seconds")
    return parser


def run_cli(args):
    proposal = plan(args.source_model, args.config, args.output)
    if args.action == "plan":
        return proposal
    if args.action == "preflight":
        return preflight(proposal)
    return execute(proposal, timeout=args.timeout)


def main(argv=None):
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_cli(sub)
    args = parser.parse_args(["retarget", *(sys.argv[1:] if argv is None else argv)])
    try:
        result = run_cli(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"Retarget error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if args.action == "preflight" and not result["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
