#!/usr/bin/env python3
"""Print the saved trainer ``global_step`` of a .pt checkpoint without importing Torch.

The checkpoint is a zip archive whose ``data.pkl`` pickle contains the trainer state; the
integer following the ``global_step`` key is scanned from the pickle opcodes. Exit 0 and print
the step on success; exit 1 and print ``unknown`` otherwise. Safe to run on login/CPU nodes.
"""
from __future__ import annotations

import io
import pickletools
import sys
import zipfile

_INT_OPS = {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"}


def read_global_step(path: str) -> int | None:
    # streamed: the pickle is read from the archive entry as a file object and the opcode generator stops at the
    # integer that follows the ``global_step`` key - nothing of the (large) trainer-state pickle is materialized
    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith("data.pkl")]
            if not names:
                return None
            with archive.open(names[0]) as handle:
                pending = 0
                for op, arg, _pos in pickletools.genops(io.BufferedReader(handle)):
                    if arg == "global_step":
                        pending = 5  # (re)armed at every occurrence, as the former full-window scan did
                        continue
                    if pending:
                        pending -= 1
                        if op.name in _INT_OPS:
                            return int(arg)
    except (OSError, zipfile.BadZipFile, ValueError, EOFError):
        return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: ckpt_global_step.py <checkpoint.pt>", file=sys.stderr)
        return 2
    step = read_global_step(argv[1])
    if step is None:
        print("unknown")
        return 1
    print(step)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
