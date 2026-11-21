"""Source checkout entry point for batched surface-grasp proposals."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.surface_grasp import main

if __name__ == "__main__":
    main()
