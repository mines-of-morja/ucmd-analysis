"""Green–Kubo Onsager coefficients (A1.5, design/A1.5-green-kubo.md of UnitcellMD, option A).

Known answers: Ornstein–Uhlenbeck velocities with component variance σ² and correlation time τ have
D = σ²τ, so N independent ions give L = N D/(k_B T V). Positions are the exact integral of the same
process, so the Einstein form must give the same number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ucmd_analysis import cli
from ucmd_analysis import constants as c
from ucmd_analysis.green_kubo import onsager_green_kubo, running_integral
from ucmd_analysis.io import Trajectory, read_h5md, system_from_json
from ucmd_analysis.msd import msd_series

T_K, BOX_NM, N, FRAMES, DT_PS = 300.0, 4.0, 8, 60000, 0.02
TAU_PS, VAR = 0.2, 0.04  # ps, nm²/ps² per component
PLATEAU = (1.0, 1.6)  # ps: 5 to 8 τ, fixed here before any comparison


def ou_path(rng, n: int, var: float, tau: float, frames: int, dt: float):
    """Exact Ornstein–Uhlenbeck velocities and their exact integral (positions), (frames, n, 3)."""
    a = np.exp(-dt / tau)
    cvv = var * (1 - a * a)
    cxx = var * tau * tau * (2 * dt / tau - 3 + 4 * a - a * a)
    cxv = var * tau * (1 - a) ** 2
    l11 = np.sqrt(cvv)
    l21 = cxv / l11
    l22 = np.sqrt(cxx - l21 * l21)
    g = rng.normal(size=(frames, 2, n, 3))
    v, x = np.empty((frames, n, 3)), np.empty((frames, n, 3))
    v[0], x[0] = np.sqrt(var) * g[0, 0], 0.0
    for k in range(1, frames):
        x[k] = x[k - 1] + v[k - 1] * tau * (1 - a) + l21 * g[k, 0] + l22 * g[k, 1]
        v[k] = a * v[k - 1] + l11 * g[k, 0]
    return v, x


def _doc() -> dict:
    def one(element: str, q: int, mass: float) -> dict:
        atom = {"name": "X", "element": element, "mass_amu": mass, "charge_e": 0.75 * q}
        return {"formal_charge_e": q, "atoms": [atom], "bonds": []}

    return {
        "molecule_types": {"Li": one("Li", 1, 7.0), "A": one("F", -1, 145.0),
                           "S": one("Xe", 0, 1.0e9)},
        "components": [{"molecule_type": "Li", "count": N}, {"molecule_type": "A", "count": N},
                       {"molecule_type": "S", "count": 1}],
    }  # fmt: skip


def _traj(seed: int, shared: float = 0.0, drift: float = 0.0) -> Trajectory:
    """Ions with OU velocities; `shared` is the variance of a part common to each Li–A pair. One
    very heavy static particle pins the barycentre, like the solvent of an electrolyte."""
    rng = np.random.default_rng(seed)
    v, x = ou_path(rng, 2 * N, VAR, TAU_PS, FRAMES, DT_PS)
    if shared:
        vs, xs = ou_path(rng, N, shared, TAU_PS, FRAMES, DT_PS)
        v = v + np.concatenate([vs, vs], axis=1)
        x = x + np.concatenate([xs, xs], axis=1)
    t = np.arange(FRAMES) * DT_PS
    zero = np.zeros((FRAMES, 1, 3))
    v = np.concatenate([v, zero], axis=1) + drift
    x = np.concatenate([x, zero], axis=1) + drift * t[:, None, None]
    return Trajectory(system_from_json(_doc()), np.arange(FRAMES), t, x,
                      np.tile(np.eye(3) * BOX_NM, (FRAMES, 1, 1)), {},
                      velocities_nm_per_ps=v)  # fmt: skip


@pytest.fixture(scope="module")
def independent() -> Trajectory:
    return _traj(3)


def _l_exact(var: float) -> float:
    d = var * TAU_PS * c.NM2_PER_PS_IN_M2_PER_S
    return N * d / (c.BOLTZMANN_J_PER_K * T_K * (BOX_NM * c.NM_IN_M) ** 3)


def test_independent_ions_give_the_analytic_value_in_both_forms(independent: Trajectory) -> None:
    out = onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=PLATEAU, max_lag_ps=2.0)
    exact = _l_exact(VAR)
    for form in ("green_kubo", "einstein_same_window"):
        for k in ("L_pp", "L_mm"):
            got = out[form][k]
            assert abs(got["mean"] - exact) < 3 * got["sem"] + 0.02 * exact, (form, k)
        pm = out[form]["L_pm"]
        assert abs(pm["mean"]) < 3 * pm["sem"] + 0.02 * exact
        assert out[form]["t_plus_COM"]["mean"] == pytest.approx(0.5, abs=0.15)
    # paired by block over the same lags: the two forms agree far better than either agrees with
    # the exact value (edge terms of the finite blocks are what is left)
    for k in ("L_pp", "L_mm", "L_pm"):
        assert abs(out["difference"][k]["mean"]) < 0.01 * exact
    # the default fit window (24–120 ps of a 240 ps block) is noisy, but not biased
    ein = out["einstein"]["L_pp"]
    assert abs(ein["mean"] - exact) < 3 * ein["sem"]


def test_pairs_sharing_motion_give_l_pm(independent: Trajectory) -> None:
    shared = 0.04
    out = onsager_green_kubo(_traj(4, shared=shared), T_K, "Li", "A",
                             plateau_ps=PLATEAU, max_lag_ps=2.0)  # fmt: skip
    exact = _l_exact(shared)
    for form in ("green_kubo", "einstein_same_window"):
        pm = out[form]["L_pm"]
        assert pm["mean"] > 0
        assert abs(pm["mean"] - exact) < 3 * pm["sem"] + 0.02 * exact
        # κ = e²(L₊₊ + L₋₋ − 2L₊₋): the shared part cancels
        kappa = out[form]["kappa_S_per_m"]
        e2 = c.ELEMENTARY_CHARGE_C**2
        assert abs(kappa["mean"] - 2 * e2 * _l_exact(VAR)) < 3 * kappa["sem"] + 0.05 * e2 * exact


def test_the_integral_is_half_the_slope_of_the_msd() -> None:
    """I(t) = ½ d/dt MSD for one series; a finite sample adds edge terms of order lag/length."""
    rng = np.random.default_rng(9)
    v, x = ou_path(rng, 1, VAR, TAU_PS, 200000, DT_PS)
    lags = 80
    i_t = running_integral(v[:, 0], v[:, 0], DT_PS, lags)
    msd = msd_series(x[:, 0])[: lags + 2]
    half_slope = 0.5 * (msd[2:] - msd[:-2]) / (2 * DT_PS)
    assert np.allclose(i_t[10:lags], half_slope[9 : lags - 1], rtol=0.01)


def test_a_common_drift_changes_nothing() -> None:
    a = onsager_green_kubo(_traj(5), T_K, "Li", "A", plateau_ps=PLATEAU, max_lag_ps=2.0)
    b = onsager_green_kubo(_traj(5, drift=0.7), T_K, "Li", "A",
                           plateau_ps=PLATEAU, max_lag_ps=2.0)  # fmt: skip
    for k in ("L_pp", "L_mm", "L_pm", "kappa_S_per_m"):
        assert b["green_kubo"][k]["mean"] == pytest.approx(a["green_kubo"][k]["mean"], rel=1e-6)


def test_stride_study_and_plateau_diagnostics(independent: Trajectory) -> None:
    out = onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=PLATEAU, max_lag_ps=2.0)
    s = out["stride_study"]["L_pp"]
    assert set(s) == {"1", "2", "4"}
    assert s["1"] == pytest.approx(out["green_kubo"]["L_pp"]["mean"], rel=1e-12)
    # trapezoid on exp(−t/τ): the bias grows as (dt/τ)²/12, 1.3 % at stride 4 (dt/τ = 0.4)
    assert s["4"] > s["1"]
    run = out["running"]["L_pp"]
    assert len(run["lag_ps"]) == len(run["mean"]) == len(run["sem"])
    d = out["plateau"]["L_pp"]
    assert set(d) >= {"drift", "flat"} and isinstance(d["flat"], bool)
    assert out["_meta"]["plateau_ps"] == list(PLATEAU)


def test_a_window_on_the_rising_flank_is_not_flat(independent: Trajectory) -> None:
    out = onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=(0.02, 0.4), max_lag_ps=2.0)
    assert out["plateau"]["L_pp"]["flat"] is False


def test_refusals(independent: Trajectory) -> None:
    good = {"plateau_ps": PLATEAU, "max_lag_ps": 2.0}
    bare = Trajectory(independent.system, independent.steps, independent.times_ps,
                      independent.positions_nm, independent.boxes_nm, {})  # fmt: skip
    with pytest.raises(ValueError, match="velocities"):
        onsager_green_kubo(bare, T_K, "Li", "A", **good)
    with pytest.raises(ValueError, match="plateau"):
        onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=(1.0, 3.0), max_lag_ps=2.0)
    with pytest.raises(ValueError, match="plateau"):
        onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=(1.5, 1.0), max_lag_ps=2.0)
    with pytest.raises(ValueError, match="too short"):
        onsager_green_kubo(independent, T_K, "Li", "A", plateau_ps=(60.0, 90.0), max_lag_ps=100.0)
    with pytest.raises(ValueError, match="neutral"):
        onsager_green_kubo(independent, T_K, "S", "A", **good)


def _write(path: Path, traj: Trajectory, velocity_steps=None) -> None:
    import h5py

    with h5py.File(path, "w") as fh:
        for name, value in (
            ("position", traj.positions_nm),
            ("velocity", traj.velocities_nm_per_ps),
        ):
            g = fh.create_group(f"particles/all/{name}")
            steps = traj.steps if name == "position" or velocity_steps is None else velocity_steps
            g["step"], g["time"] = steps, np.asarray(steps) * DT_PS
            g["value"] = value[: len(steps)]
        e = fh.create_group("particles/all/box/edges")
        e["step"], e["time"], e["value"] = traj.steps, traj.times_ps, traj.boxes_nm
        fh.create_group("parameters/ucmd")["system_json"] = json.dumps(traj.system.doc)


def test_reader_and_cli(tmp_path: Path, independent: Trajectory, capsys) -> None:
    n = 6000
    short = Trajectory(independent.system, independent.steps[:n], independent.times_ps[:n],
                       independent.positions_nm[:n], independent.boxes_nm[:n], {},
                       velocities_nm_per_ps=independent.velocities_nm_per_ps[:n])  # fmt: skip
    path = tmp_path / "v.h5"
    _write(path, short)
    assert read_h5md(path).velocities_nm_per_ps is None  # not read unless asked for
    got = read_h5md(path, start=5, velocities=True)
    assert np.array_equal(got.velocities_nm_per_ps, short.velocities_nm_per_ps[5:])
    args = ["onsager-gk", str(path), "--temperature-K", "300", "--anion", "A",
            "--plateau-ps", "1.0", "1.6", "--max-lag-ps", "2"]  # fmt: skip
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["green_kubo"]["L_pp"]["mean"] > 0 and out["skip_ps"] == 0.0
    with pytest.raises(SystemExit):  # the window has no default
        cli.main(args[:6])
    other = tmp_path / "w.h5"  # velocities at other steps than positions
    _write(other, short, velocity_steps=np.arange(n) * 2)
    with pytest.raises(ValueError, match="same steps"):
        read_h5md(other, velocities=True)
    none = tmp_path / "n.h5"
    import h5py

    _write(none, short)
    with h5py.File(none, "a") as fh:
        del fh["particles/all/velocity"]
    with pytest.raises(ValueError, match="velocities = true"):
        read_h5md(none, velocities=True)
