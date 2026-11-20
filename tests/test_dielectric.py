"""Static dielectric constant from dipole fluctuations (calculation plan A7, design E2.9 §3).

Known answer: N independent, randomly oriented rigid dipoles of moment μ have ⟨M⟩ = 0 and
⟨M²⟩ = Nμ², so ε = 1 + Nμ²/(3ε₀Vk_BT) exactly in expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ucmd_analysis import constants as c
from ucmd_analysis.dielectric import dielectric, total_dipole
from ucmd_analysis.io import Trajectory, system_from_json

Q, D = 0.5, 0.1  # e, nm: μ = 0.05 e nm per dimer
BOX = 2.0  # nm
T = 300.0


def _doc(n_dimers: int, n_ions: int = 0) -> dict:
    types = {
        "D": {"formal_charge_e": 0, "atoms": [
            {"element": "X", "mass_amu": 10.0, "charge_e": Q},
            {"element": "X", "mass_amu": 10.0, "charge_e": -Q}], "bonds": [{"atoms": [0, 1]}]},
        "Na": {"formal_charge_e": 1, "atoms": [
            {"element": "Na", "mass_amu": 23.0, "charge_e": 0.8}], "bonds": []},
    }  # fmt: skip
    comps = [{"molecule_type": "D", "count": n_dimers}]
    if n_ions:
        comps.append({"molecule_type": "Na", "count": n_ions})
    return {"molecule_types": types, "components": comps}


def _trajectory(rng: np.random.Generator, n: int, frames: int, n_ions: int = 0) -> Trajectory:
    centres = (
        rng.uniform(0.0, BOX, (frames, n, 1, 3)) + rng.integers(-3, 3, (frames, n, 1, 3)) * BOX
    )
    u = rng.normal(size=(frames, n, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    pos = np.concatenate([centres + 0.5 * D * u[:, :, None], centres - 0.5 * D * u[:, :, None]],
                         axis=2).reshape(frames, 2 * n, 3)  # fmt: skip
    if n_ions:  # ions diffuse far: their current must not enter ε
        ions = np.cumsum(rng.normal(0.0, 0.3, (frames, n_ions, 3)), axis=0)
        pos = np.concatenate([pos, ions], axis=1)
    return Trajectory(
        system_from_json(_doc(n, n_ions)),
        np.arange(frames),
        np.arange(frames, dtype=float),
        pos,
        np.tile(np.eye(3) * BOX, (frames, 1, 1)),
        {},
    )


def _expected(n: int) -> float:
    kt = c.BOLTZMANN_KJMOL_PER_K * T
    return 1.0 + n * (Q * D) ** 2 / (3.0 * c.EPSILON0_E2_PER_KJMOL_NM * BOX**3 * kt)


def test_epsilon0_from_codata() -> None:
    # 1/(4πε₀) is the Coulomb constant 138.935 kJ mol⁻¹ nm e⁻² (GROMACS manual, ONE_4PI_EPS0)
    assert 1.0 / (4.0 * np.pi * c.EPSILON0_E2_PER_KJMOL_NM) == pytest.approx(138.935458, rel=1e-8)


def test_independent_dipoles_give_the_known_epsilon() -> None:
    rng = np.random.default_rng(1)
    traj = _trajectory(rng, n=60, frames=8000)
    out = dielectric(traj, T, blocks=5)
    assert out["epsilon"] == pytest.approx(_expected(60), rel=0.05)  # ⟨M²⟩ to ≈ 1.6 % (1σ)
    assert abs(out["epsilon"] - _expected(60)) < 4 * out["epsilon_blocks"]["sem"]
    assert out["molecule_types"] == ["D"]


def test_whole_molecules_in_any_image_give_the_same_dipole() -> None:
    rng = np.random.default_rng(2)
    traj = _trajectory(rng, n=10, frames=3)
    shifted = Trajectory(traj.system, traj.steps, traj.times_ps, traj.positions_nm + 7.3,
                         traj.boxes_nm, {})  # fmt: skip
    assert np.allclose(total_dipole(traj, ["D"]), total_dipole(shifted, ["D"]), atol=1e-12)


def test_ions_are_left_out_and_a_charged_selection_is_refused() -> None:
    rng = np.random.default_rng(3)
    traj = _trajectory(rng, n=60, frames=8000, n_ions=5)
    out = dielectric(traj, T, blocks=5)
    assert out["molecule_types"] == ["D"]
    assert out["epsilon"] == pytest.approx(_expected(60), rel=0.05)
    with pytest.raises(ValueError, match="not neutral"):
        total_dipole(traj, ["Na"])
