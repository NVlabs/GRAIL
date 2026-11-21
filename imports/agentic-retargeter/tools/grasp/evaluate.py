"""Source checkout entry point for common mounted grasp evaluation."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.grasp_evaluation import main

if __name__ == "__main__":
    main()
