"""Checkout entry for authored route and batched transition repair."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))[
    "bootstrap"
]()
from agentic_retargeter.transition_repair import main

if __name__ == "__main__":
    raise SystemExit(main())
