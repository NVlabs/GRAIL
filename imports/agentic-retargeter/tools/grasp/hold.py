"""Source checkout entry point for shared-grasp mounted native proposals."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.shared_grasp import main

if __name__ == "__main__":
    main()
