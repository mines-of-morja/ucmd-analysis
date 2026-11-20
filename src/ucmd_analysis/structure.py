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


def first_minimum(r: np.ndarray, g: np.ndarray, smooth_nm: float = 0.025) -> float:
    """The first minimum of g(r) after its first peak (nm), robust to bin-to-bin noise.

    - g is smoothed by a moving average over `smooth_nm`. The result can shift by up to half
      that width from the minimum of the unsmoothed curve.
    - The peak is the maximum of the smoothed g.
    - The minimum is the first local minimum after it that lies below the midpoint between the
      peak and 1, so a noise dip on a broad peak does not count. In a flat valley this is its
      start, so the cutoff does not slide outward into the second shell.
    - Raises ValueError if there is none within the range of r.
    """
    w = max(1, round(smooth_nm / (r[1] - r[0])))
    gs = np.convolve(np.pad(g, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, "valid")
    peak = int(np.argmax(gs))
    deep = 1.0 + 0.5 * (gs[peak] - 1.0)
    for k in range(peak + 1, len(gs) - 1):
        if gs[k] <= gs[k + 1] and gs[k] < deep:
            return float(r[k])
    raise ValueError("no first minimum of g(r) within the range; give the cutoff explicitly")


def coordination(r: np.ndarray, g: np.ndarray, density_b_per_nm3: float,
                 r_cut_nm: float | None = None) -> tuple[float, float]:  # fmt: skip
    """(n, r_cut): the number of b within r_cut of an a, n = 4πρ_b ∫₀^r_cut g r² dr. Without
    r_cut, `first_minimum` is used."""
    if r_cut_nm is None:
        r_cut_nm = first_minimum(r, g)
    dr = r[1] - r[0]
    mask = r <= r_cut_nm
    return float(4.0 * np.pi * density_b_per_nm3 * np.sum(g[mask] * r[mask] ** 2) * dr), r_cut_nm
