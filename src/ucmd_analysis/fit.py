"""Slope of a correlation over a lag window, shared by the transport modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitWindow:
    """Fit window as fractions of the longest lag of a block."""

    start: float = 0.1
    stop: float = 0.5


def slope(lags_ps: np.ndarray, y: np.ndarray, window: FitWindow) -> float:
    n = len(lags_ps)
    lo, hi = max(1, int(window.start * n)), max(2, int(window.stop * n))
    return float(np.polyfit(lags_ps[lo:hi], y[lo:hi], 1)[0])
