"""Compatibility import for :mod:`grail.optimization.hoi_stationary`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_stationary as _implementation

sys.modules[__name__] = _implementation
