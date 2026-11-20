"""The first minimum of g(r) (the default contact cutoff) must not be fooled by noise.

The two shapes come from the 1 M LiPF₆ EC:EMC run (2026-11-20):
- Li–P: a broad first peak (0.28–0.40 nm) with bin-to-bin noise of ±0.4 on top; the minimum is
  near 0.45 nm. A rule that stops at the first rise after the global maximum ends inside the
  peak (it gave 0.3275 nm).
- Li–O(carbonyl): a sharp peak at 0.2 nm, then a long flat valley with small noise. The cutoff
  belongs at the start of the valley, not at its noisiest low point further out.
"""

from __future__ import annotations

import numpy as np
import pytest

from ucmd_analysis.structure import coordination, first_minimum

R = np.arange(0.0025, 0.8, 0.005)  # the bins of the CLI: 160 over 0.8 nm


def _old_rule(r: np.ndarray, g: np.ndarray) -> float:
    peak = int(np.argmax(g))
    return float(r[peak + int(np.argmax(np.diff(g[peak:]) > 0))])


def test_broad_noisy_peak() -> None:
    smooth = (
        1.0
        + 1.6 * np.exp(-(((R - 0.34) / 0.045) ** 2))
        - 0.6 * np.exp(-(((R - 0.455) / 0.03) ** 2))
    )
    smooth[R < 0.27] = 0.0
    noise = np.where((R > 0.29) & (R < 0.38), 0.4 * (-1.0) ** np.arange(len(R)), 0.0)
    g = smooth + noise
    true_min = R[np.argmin(np.where(R > 0.38, smooth, np.inf))]
    assert _old_rule(R, g) < 0.38  # the old rule is fooled: the test can tell the two apart
    assert first_minimum(R, g) == pytest.approx(true_min, abs=0.015)


def test_sharp_peak_then_flat_valley() -> None:
    g = 60.0 * np.exp(-(((R - 0.205) / 0.012) ** 2))
    valley = (R > 0.25) & (R < 0.42)
    g = np.where(valley, np.maximum(g, 0.05), g)
    g += np.where(valley, 0.01 * np.sin(97.0 * R), 0.0)
    g = np.where(R >= 0.42, np.minimum(1.0, 0.05 + 4.0 * (R - 0.42)), g)
    r_cut = first_minimum(R, g)
    assert 0.24 < r_cut < 0.30  # start of the valley, not its far end
    n, _ = coordination(R, g, 1.0, r_cut_nm=r_cut)
    n_wide, _ = coordination(R, g, 1.0, r_cut_nm=0.40)
    assert n == pytest.approx(n_wide, rel=0.02)  # nothing of the shell is left out


def test_no_minimum_is_an_error() -> None:
    g = np.clip((R - 0.2) * 5.0, 0.0, None)  # rises to the edge: the shell never closes
    with pytest.raises(ValueError, match="no first minimum"):
        first_minimum(R, g)
