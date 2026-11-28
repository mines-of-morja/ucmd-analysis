"""Uncertainties (PLAN.md A1.6): block averages and replica statistics."""

from __future__ import annotations

import numpy as np


def mean_sem(values) -> tuple[float, float]:
    """Mean and standard error of the mean of independent estimates (blocks or replicas)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def blocks(n_frames: int, n_blocks: int) -> list[slice]:
    """Consecutive, equally long frame ranges."""
    size = n_frames // n_blocks
    return [slice(k * size, (k + 1) * size) for k in range(n_blocks)]
