"""Compatibility import for :mod:`grail.optimization.hoi_depth_alignment`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_depth_alignment as _implementation

sys.modules[__name__] = _implementation
