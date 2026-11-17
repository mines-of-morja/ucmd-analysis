"""A1 core on synthetic data with known answers (PLAN.md A1 ✓ criteria)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ucmd_analysis import constants as c
from ucmd_analysis.io import read_h5md, system_from_json
from ucmd_analysis.msd import cross_msd, msd_series, self_msd
from ucmd_analysis.structure import coordination, rdf
from ucmd_analysis.transport import FitWindow, transport


def direct_msd(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.array(
        [np.mean(np.sum((x[m:] - x[: n - m]).reshape(n - m, -1) ** 2, axis=1)) for m in range(n)]
    )


def test_fft_msd_equals_the_direct_sum_and_the_ballistic_case() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(300, 4, 3)), axis=0)
    assert np.allclose(msd_series(x), direct_msd(x))
    assert np.allclose(self_msd(x), direct_msd(x) / 4)
    t = np.arange(50.0)
    v = np.array([0.3, -0.2, 0.1])
    ballistic = t[:, None] * v
    assert np.allclose(msd_series(ballistic), (t * np.linalg.norm(v)) ** 2)
    a, b = x[:, 0], x[:, 1]
    direct = np.array([np.mean(np.sum((a[m:] - a[:-m or None]) * (b[m:] - b[:-m or None]), 1))
                       for m in range(len(a))])  # fmt: skip
    assert np.allclose(cross_msd(a, b), direct)


def spec_doc(n_ions: int, box: float) -> dict:
    def ion(q: int, mass: float) -> dict:
        atom = {"name": "X", "element": "Li" if q > 0 else "F", "mass_amu": mass}
        return {"formal_charge_e": q, "atoms": [{**atom, "charge_e": 0.8 * q}], "bonds": []}

    return {
        "atom_count": 2 * n_ions,
        "cell_nm": [[box, 0, 0], [0, box, 0], [0, 0, box]],
        "molecule_types": {"Li": ion(1, 7.0), "A": ion(-1, 145.0)},
        "components": [
            {"molecule_type": "Li", "count": n_ions},
            {"molecule_type": "A", "count": n_ions},
        ],
    }


def write_h5md(path: Path, doc: dict, times: np.ndarray, x: np.ndarray, box: float) -> None:
    import h5py

    with h5py.File(path, "w") as fh:
        p = fh.create_group("particles/all/position")
        p["step"] = np.arange(len(times))
        p["time"] = times
        p["value"] = x
        e = fh.create_group("particles/all/box/edges")
        e["step"] = np.arange(len(times))
        e["time"] = times
        e["value"] = np.tile(np.eye(3) * box, (len(times), 1, 1))
        u = fh.create_group("parameters/ucmd")
        u["system_json"] = json.dumps(doc)
        u.attrs["engine_version"] = "synthetic"


def test_uncorrelated_random_walkers_give_nernst_einstein(tmp_path: Path) -> None:
    """Independent Brownian ions: L₊₋ → 0, κ = κ_NE, D as set (in the barycentric frame)."""
    rng = np.random.default_rng(7)
    n, frames, dt, box = 200, 4000, 0.1, 5.0
    d_li, d_a = 0.002, 0.0005  # nm²/ps
    steps = np.concatenate(
        [rng.normal(0, np.sqrt(2 * d_li * dt), (frames, n, 3)),
         rng.normal(0, np.sqrt(2 * d_a * dt), (frames, n, 3))], axis=1,
    )  # fmt: skip
    x = np.cumsum(steps, axis=0)
    doc = spec_doc(n, box)
    write_h5md(tmp_path / "t.h5", doc, np.arange(frames) * dt, x, box)
    traj = read_h5md(tmp_path / "t.h5")
    out = transport(traj, 300.0, "Li", "A", n_blocks=5, window=FitWindow(0.05, 0.3))
    to_nm2ps = 1 / c.NM2_PER_PS_IN_M2_PER_S
    # barycentric frame: the light Li barely moves the centre of mass; tolerance 5 %
    assert out["D_Li_m2_per_s"]["mean"] * to_nm2ps == pytest.approx(d_li, rel=0.05)
    assert out["D_A_m2_per_s"]["mean"] * to_nm2ps == pytest.approx(d_a, rel=0.07)
    assert out["ionicity"]["mean"] == pytest.approx(1.0, abs=0.15)
    lpm = out["L_pm"]
    assert abs(lpm["mean"]) < 3 * lpm["sem"] + 0.05 * out["L_pp"]["mean"]
    assert 0.0 < out["t_plus_COM"]["mean"] < 1.0
    scaled = transport(traj, 300.0, "Li", "A", charges="scaled", window=FitWindow(0.05, 0.3))
    ratio = scaled["kappa_S_per_m"]["mean"] / out["kappa_S_per_m"]["mean"]
    assert ratio == pytest.approx(0.8**2, rel=1e-9)  # κ ∝ z²: ζ² for scaled charges


def test_ideal_gas_rdf_is_one_and_coordination_counts_neighbours() -> None:
    rng = np.random.default_rng(3)
    box = 4.0
    x = rng.uniform(0, box, (20, 600, 3))
    boxes = np.tile(np.eye(3) * box, (20, 1, 1))
    idx = np.arange(600)
    _r, g = rdf(x, boxes, idx[:300], idx[300:], r_max_nm=1.5, bins=30)
    assert np.allclose(g[5:], 1.0, atol=0.08)
    # a shell: 12 b atoms at 0.3 nm around each a gives n = 12 within 0.4 nm
    density = 1.0
    rr = np.linspace(0.005, 1.0, 100)
    gg = np.zeros_like(rr)
    k = np.argmin(abs(rr - 0.3))
    gg[k] = 12 / (4 * np.pi * density * rr[k] ** 2 * (rr[1] - rr[0]))
    n, _ = coordination(rr, gg, density, r_cut_nm=0.4)
    assert n == pytest.approx(12.0, rel=1e-9)


def test_carbonyl_oxygen_selection_by_bond_count() -> None:
    doc = {
        "molecule_types": {
            "EC": {"formal_charge_e": 0, "atoms": [
                {"element": "O", "mass_amu": 16.0}, {"element": "C", "mass_amu": 12.0},
                {"element": "O", "mass_amu": 16.0}],
                "bonds": [{"atoms": [0, 1]}, {"atoms": [1, 2]}, {"atoms": [2, 1]}]},
        },
        "components": [{"molecule_type": "EC", "count": 2}],
    }  # fmt: skip
    sys = system_from_json(doc)
    assert sys.atoms_where("EC", "O").tolist() == [0, 2, 3, 5]
    assert sys.atoms_where("EC", "O", bonded=1).tolist() == [0, 3]
