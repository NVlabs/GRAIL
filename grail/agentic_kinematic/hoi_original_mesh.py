"""Compatibility import for :mod:`grail.optimization.hoi_original_mesh`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_original_mesh as _implementation

sys.modules[__name__] = _implementation
