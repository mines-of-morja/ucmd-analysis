"""Known answers for analyze.py: run explicitly, `pytest validation/a17-lj`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analyze as a
from make_system import SIGMA_NM

from ucmd_analysis.structure_factor import SkSeries, reciprocal_vectors

# symmetric mixture with rho kT kappa = 0.122 and an ideal difference mode: S0_AB = -0.439
S0 = np.array([[0.561, -0.439], [-0.439, 0.561]])


def series(frames: int = 5000, seed: int = 1) -> SkSeries:
    """Gaussian modes with S(k)^-1 = S0^-1 + k2 L on the box of the run (L = 20 sigma)."""
    _, k2 = reciprocal_vectors(np.eye(3) * 20 * SIGMA_NM, 2 * np.pi * 0.3 / SIGMA_NM)
    rng = np.random.default_rng(seed)
    lam = np.eye(2) * 0.5 * SIGMA_NM**2
    rho = np.empty((frames, 2, len(k2)), dtype=complex)
    for j, q in enumerate(k2):
        chol = np.linalg.cholesky(np.linalg.inv(np.linalg.inv(S0) + q * lam) * 2000 / 2)
        z = rng.normal(size=(frames, 2)) + 1j * rng.normal(size=(frames, 2))
        rho[:, :, j] = z @ chol.T
    return SkSeries(rho, k2, np.array([2000.0, 2000.0]))


def test_matrix_form_recovers_the_prescribed_limit():
    for f, want in ((a.s_ab, -0.439), (a.compressibility, 0.122), (a.dmu, 1.0)):
        value, err = a.with_jackknife(series(), 0.2, "matrix", f)
        assert abs(value - want) < 4 * err
        assert err < 0.02 * abs(want)


def test_jackknife_error_matches_the_scatter_of_independent_series():
    got = [a.with_jackknife(series(400, s), 0.2, "matrix", a.s_ab) for s in range(24)]
    scatter = np.std([g[0] for g in got], ddof=1)
    mean_error = np.mean([g[1] for g in got])
    assert mean_error == pytest.approx(scatter, rel=0.4)


def test_verdict_follows_the_criterion():
    def npt(value, err, ok=True):
        return {"S0_AB": {"scalar_0.2": (value, err)}, "block_long_enough": ok}

    assert a.verdict(npt(-0.449, 0.004))["verdict"] == "pass"
    assert a.verdict(npt(-0.451, 0.004))["verdict"] == "FAIL"
    assert a.verdict(npt(-0.427, 0.004))["verdict"] == "FAIL"
    assert a.verdict(npt(-0.439, 0.004, ok=False))["verdict"].startswith("not judged")
