"""Compatibility import for :mod:`grail.optimization.hoi_depth_cache`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_depth_cache as _implementation

sys.modules[__name__] = _implementation
