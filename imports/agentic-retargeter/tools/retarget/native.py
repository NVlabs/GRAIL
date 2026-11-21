"""Thin entry point for a validated SOMA run → native robot handoff."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.retarget_native_cli import main

if __name__ == "__main__":
    main()
