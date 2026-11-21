"""Repository entry point for foreground, timed case workflows."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.batch_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
