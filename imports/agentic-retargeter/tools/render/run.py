"""Render a bound native sequence in foreground Blender workers."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.render_sequence import main

if __name__ == "__main__":
    main()
