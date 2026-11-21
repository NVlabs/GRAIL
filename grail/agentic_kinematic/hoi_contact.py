"""Compatibility import for :mod:`grail.optimization.hoi_contact`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_contact as _implementation

sys.modules[__name__] = _implementation
