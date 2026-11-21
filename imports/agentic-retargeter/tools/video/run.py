"""Source-checkout entry point; implementation lives directly in src/."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[2]
runpy.run_path(str(root / "src/_bootstrap.py"))["bootstrap"]()

from agentic_retargeter.video_cli import main

if __name__ == "__main__":
    main()
