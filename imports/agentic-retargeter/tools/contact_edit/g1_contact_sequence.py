"""Thin launcher; implementation lives in src/contact_edit."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/_bootstrap.py"))["bootstrap"]()
from agentic_retargeter.contact_edit.experimental.g1_contact_sequence import *  # compatibility for existing callers

if __name__ == "__main__":
    main()
