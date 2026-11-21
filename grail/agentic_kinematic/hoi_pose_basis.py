"""Compatibility import for :mod:`grail.optimization.hoi_pose_basis`. Use the canonical module in new code."""

import sys

from grail.optimization import hoi_pose_basis as _implementation

sys.modules[__name__] = _implementation
