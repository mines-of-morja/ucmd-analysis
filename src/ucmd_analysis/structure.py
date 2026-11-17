"""Radial distribution functions and coordination numbers (PLAN.md A1.3).

Distances use the minimum image in any (reduced, triclinic) cell, through fractional
coordinates. g(r) is normalized by the ideal-gas pair density of the frame's volume.
"""

from __future__ import annotations

import numpy as np


def rdf(positions: np.ndarray, boxes: np.ndarray, a: np.ndarray, b: np.ndarray, *,
        r_max_nm: float, bins: int = 200) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(r, g(r)) between atom groups a and b over all frames; pairs i == j are excluded."""
    edges = np.linspace(0.0, r_max_nm, bins + 1)
    hist = np.zeros(bins)
    same = set(a.tolist()) & set(b.tolist())
    norm = 0.0
    for x, h in zip(positions, boxes, strict=True):
        hinv = np.linalg.inv(h)
        xa, xb = x[a], x[b]
        for k in range(0, len(a), 64):  # chunks keep memory small
            d = xa[k : k + 64, None, :] - xb[None, :, :]
            s = d @ hinv
            s -= np.round(s)
            r = np.linalg.norm(s @ h, axis=-1)
            if same:
                r[a[k : k + 64, None] == b[None, :]] = np.inf
            hist += np.histogram(r, bins=edges)[0]
        n_pairs = len(a) * len(b) - len(same)
        norm += n_pairs / abs(np.linalg.det(h))
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    r = 0.5 * (edges[1:] + edges[:-1])
    return r, hist / (norm * shell)


def coordination(r: np.ndarray, g: np.ndarray, density_b_per_nm3: float,
                 r_cut_nm: float | None = None) -> tuple[float, float]:  # fmt: skip
    """(n, r_cut): the number of b within r_cut of an a, n = 4πρ_b ∫₀^r_cut g r² dr. Without
    r_cut, the first minimum of g after its first peak is used."""
    if r_cut_nm is None:
        peak = int(np.argmax(g))
        after = g[peak:]
        k = peak + int(np.argmax(np.diff(after) > 0)) if np.any(np.diff(after) > 0) else len(g) - 1
        r_cut_nm = float(r[k])
    dr = r[1] - r[0]
    mask = r <= r_cut_nm
    return float(4.0 * np.pi * density_b_per_nm3 * np.sum(g[mask] * r[mask] ** 2) * dr), r_cut_nm
