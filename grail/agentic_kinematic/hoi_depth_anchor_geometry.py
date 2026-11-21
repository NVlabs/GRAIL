"""Compatibility import for :mod:`grail.optimization.hoi_depth_anchor_geometry`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_depth_anchor_geometry as _implementation

sys.modules[__name__] = _implementation
