"""Compatibility import for :mod:`grail.optimization.hoi_acquisition`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_acquisition as _implementation

sys.modules[__name__] = _implementation
