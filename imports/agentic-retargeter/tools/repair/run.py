"""Checkout launcher for shared motion audits and coupled window repairs."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))[
    "bootstrap"
]()
from agentic_retargeter.local_repair import main

if __name__ == "__main__":
    raise SystemExit(main())
