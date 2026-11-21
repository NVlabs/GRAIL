"""Small helpers for finite empty-window episode statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def episode_window_sum_and_count(values: Iterable[float]) -> tuple[float, int]:
    """Return a sum/count pair, using ``(0, 0)`` only for an empty window.

    Non-finite values in a non-empty window deliberately propagate through the
    sum so real numerical failures remain visible to strict log gates.
    """
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return 0.0, 0
    return float(array.sum()), int(array.size)
