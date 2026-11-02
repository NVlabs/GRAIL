#!/usr/bin/env python3
"""Compatibility shim for ``python -m grail.cli.estimate_soma_motion``."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grail.cli.estimate_soma_motion import main, sha256

if __name__ == "__main__":
    raise SystemExit(main())
