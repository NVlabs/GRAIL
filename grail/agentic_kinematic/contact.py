"""Compatibility import for :mod:`grail.core.contact`. Use the canonical module in new code."""

import sys

from grail.core import contact as _implementation

sys.modules[__name__] = _implementation
