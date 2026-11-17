"""Mean squared displacements by FFT (PLAN.md A1.4).

For a series x(t) of N frames the time-averaged MSD over all time origins is
MSD(m) = 1/(N−m) Σ_k |x(k+m) − x(k)|², computed in O(N log N) as
S1(m) − 2·S2(m) with S2 the autocorrelation (Wiener–Khinchin) and S1 a running sum
(Calandrini et al., Collection SFN 12, 201 (2011)). Frames must be equally spaced.
"""

from __future__ import annotations

import numpy as np


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Σ_k x(k)·x(k+m) / (N−m) for each lag m, summed over the trailing dimensions."""
    n = x.shape[0]
    f = np.fft.rfft(x, n=2 * n, axis=0)
    power = (f * f.conj()).real
    acf = np.fft.irfft(power, n=2 * n, axis=0)[:n]
    acf = acf.reshape(n, -1).sum(axis=1)
    return acf / (n - np.arange(n))


def msd_series(x: np.ndarray) -> np.ndarray:
    """MSD(lag) of one trajectory x of shape (frames, ..., 3), summed over all but the first
    axis (so for (frames, 3) it is |Δr|², for (frames, particles, 3) the sum over particles)."""
    n = x.shape[0]
    flat = x.reshape(n, -1)
    d = (flat**2).sum(axis=1)
    d = np.append(d, 0.0)
    q = 2.0 * d.sum()
    s1 = np.empty(n)
    for m in range(n):
        q -= d[m - 1] + d[n - m]
        s1[m] = q / (n - m)
    return s1 - 2.0 * _autocorrelation(flat)


def self_msd(x: np.ndarray) -> np.ndarray:
    """Mean over particles of the per-particle MSD; x has shape (frames, particles, 3)."""
    return msd_series(x) / x.shape[1]


def cross_msd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """⟨ΔA(lag)·ΔB(lag)⟩ over time origins for two series of shape (frames, 3)."""
    plus = msd_series(a + b)
    minus = msd_series(a - b)
    return 0.25 * (plus - minus)
