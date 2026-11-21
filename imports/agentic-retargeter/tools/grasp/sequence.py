"""Checkout entry point for coupled material holds and acquisition buffers."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.grasp_sequence import main

if __name__ == "__main__":
    raise SystemExit(main())
