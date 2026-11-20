"""Enthalpy of vaporization from a liquid and a gas-phase thermo table (calculation plan A5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ucmd_analysis import constants as c
from ucmd_analysis.cli import main
from ucmd_analysis.hvap import hvap


def _thermo(path: Path, potential: np.ndarray, dt_ps: float = 0.1) -> Path:
    t = np.arange(len(potential)) * dt_ps
    rows = "\n".join(
        f"{k}\t{ti:.6f}\t{u:.10e}" for k, (ti, u) in enumerate(zip(t, potential, strict=True))
    )
    path.write_text("# step\ttime_ps\tpotential_kjmol\n" + rows + "\n")
    return path


def test_hvap_formula_and_errors() -> None:
    rng = np.random.default_rng(1)
    n, T = 500, 298.15
    liquid = -41.0 * n + rng.normal(0.0, 50.0, 20000)  # per molecule −41 kJ/mol
    gas = 3.7 + rng.normal(0.0, 2.0, 20000)
    out = hvap(liquid, n, gas, T, n_blocks=5)
    expected = 3.7 + 41.0 + c.GAS_CONSTANT_KJ_PER_MOL_K * T
    assert out["hvap_kj_per_mol"]["mean"] == pytest.approx(expected, abs=0.05)
    sem = np.hypot(50.0 / n, 2.0) / np.sqrt(20000)  # independent samples: the true error
    assert out["hvap_kj_per_mol"]["sem"] == pytest.approx(sem, rel=0.7)
    assert out["RT_kj_per_mol"] == pytest.approx(2.4790, abs=1e-4)


def test_cli_reads_thermo_tables_and_skips_equilibration(tmp_path: Path, capsys) -> None:
    liquid = np.r_[np.full(100, 0.0), np.full(400, -100.0 * 10)]  # first 10 ps: not equilibrated
    gas = np.full(500, 2.0)
    lq = _thermo(tmp_path / "liquid.tsv", liquid)
    gs = _thermo(tmp_path / "gas.tsv", gas)
    assert main(["hvap", "--liquid", str(lq), "--molecules", "10", "--gas", str(gs),
                 "--temperature-K", "300", "--skip-ps", "10"]) == 0  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["hvap_kj_per_mol"]["mean"] == pytest.approx(
        2.0 + 100.0 + c.GAS_CONSTANT_KJ_PER_MOL_K * 300.0, rel=1e-12, abs=0
    )
    assert out["liquid_samples"] == 400
