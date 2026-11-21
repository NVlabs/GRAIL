"""Compatibility import for :mod:`grail.optimization.hoi_temporal`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_temporal as _implementation

sys.modules[__name__] = _implementation
