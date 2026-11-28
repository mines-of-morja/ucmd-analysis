"""Green–Kubo shear viscosity from the pressure-tensor columns of the thermo table (A6).

Known answer: if every independent shear component is an Ornstein–Uhlenbeck process with
variance σ² and correlation time τ, then ∫⟨P(0)P(t)⟩dt = σ²τ and η = V σ²τ/(k_B T), exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ucmd_analysis import constants as c
from ucmd_analysis.cli import main
from ucmd_analysis.viscosity import green_kubo, shear_components

V, T = 27.0, 300.0  # nm³, K
SIGMA, TAU, DT = 150.0, 0.2, 0.002  # bar, ps, ps


def _ou(rng: np.random.Generator, n: int) -> np.ndarray:
    a = np.exp(-DT / TAU)
    x = np.empty(n)
    x[0] = rng.normal(0.0, SIGMA)
    noise = rng.normal(0.0, SIGMA * np.sqrt(1 - a * a), n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + noise[k]
    return x


def _expected() -> float:
    v_m3 = V * c.NM_IN_M**3
    p_pa2 = (SIGMA * c.BAR_IN_PA) ** 2
    return v_m3 * p_pa2 * TAU * c.PS_IN_S / (c.BOLTZMANN_J_PER_K * T)


def _tensor(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """xy, xz, yz independent OU; diagonal = mean + OU so that the two traceless combinations
    (xx−yy)/2 and (yy−zz)/2 have the same statistics as the off-diagonal components."""
    d = {k: _ou(rng, n) for k in ("xy", "xz", "yz")}
    u, w = _ou(rng, n), _ou(rng, n)  # u = (xx−yy)/2, w = (yy−zz)/2
    yy = np.full(n, 1.0) + rng.normal(0.0, 1e-9, n)  # the isotropic part does not matter
    d["xx"], d["yy"], d["zz"] = yy + 2 * u, yy, yy - 2 * w
    return d


def test_ornstein_uhlenbeck_gives_the_known_viscosity() -> None:
    rng = np.random.default_rng(1)
    p = _tensor(rng, 200_000)  # 400 ps, 2000 τ
    out = green_kubo(shear_components(p), DT, V, T, n_blocks=5, max_lag_ps=2.0)
    assert out["viscosity_Pa_s"]["mean"] == pytest.approx(_expected(), rel=0.1)
    assert abs(out["viscosity_Pa_s"]["mean"] - _expected()) < 4 * out["viscosity_Pa_s"]["sem"]
    assert out["components"] == 5


def test_cli_reads_the_thermo_table(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(2)
    n = 50_000
    p = _tensor(rng, n)
    cols = ["step", "time_ps", "potential_kjmol", "volume_nm3"]
    names = [f"pressure_{k}_bar" for k in ("xx", "yy", "zz", "xy", "xz", "yz")]
    rows = np.column_stack([np.arange(n), np.arange(n) * DT, np.zeros(n), np.full(n, V)]
                           + [p[k] for k in ("xx", "yy", "zz", "xy", "xz", "yz")])  # fmt: skip
    f = tmp_path / "thermo.tsv"
    np.savetxt(f, rows, delimiter="\t", header="\t".join(cols + names), comments="# ")
    assert main(["viscosity", str(f), "--temperature-K", str(T), "--max-lag-ps", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["viscosity_mPa_s"]["mean"] == pytest.approx(_expected() * 1e3, rel=0.25)
    assert out["volume_nm3"] == pytest.approx(V)


def test_missing_columns_are_an_error(tmp_path: Path) -> None:
    f = tmp_path / "thermo.tsv"
    f.write_text("# step\ttime_ps\tpressure_bar\n0\t0.0\t1.0\n1\t0.002\t1.0\n")
    with pytest.raises(ValueError, match="pressure_tensor"):
        main(["viscosity", str(f), "--temperature-K", "300"])
