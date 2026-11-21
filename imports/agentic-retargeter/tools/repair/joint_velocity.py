"""Check full native hinge speed without constructing mesh collision queries."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))[
    "bootstrap"
]()
from agentic_retargeter.joint_velocity import main

if __name__ == "__main__":
    raise SystemExit(main())
