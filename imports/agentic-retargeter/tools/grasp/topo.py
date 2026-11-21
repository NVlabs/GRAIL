"""Source checkout entry point for Topo-inspired multi-start hand initialization."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.topo_hand_initialization import main

if __name__ == "__main__":
    main()
