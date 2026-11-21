"""Compatibility import for :mod:`grail.optimization.hoi_collision_guard`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_collision_guard as _implementation

sys.modules[__name__] = _implementation
