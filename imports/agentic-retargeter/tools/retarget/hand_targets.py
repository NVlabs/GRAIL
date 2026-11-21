"""Checkout entry point for selected free-hand sequence → SOMA targets."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.hand_sequence_handoff import main

if __name__ == "__main__":
    main()
