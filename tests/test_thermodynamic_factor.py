"""Structure factors at small k and the thermodynamic factor (A1.7, design/A1.7-tdf.md §5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ucmd_analysis import cli
from ucmd_analysis import constants as c
from ucmd_analysis.io import Trajectory, system_from_json
from ucmd_analysis.structure_factor import (
    SkSeries,
    density_modes,
    partial_sk,
    reciprocal_vectors,
    s0,
)
from ucmd_analysis.thermodynamic_factor import (
    relaxation_time_ps,
    tdf_from_fluctuations,
    thermodynamic_factor,
)

BOX = 4.0


def test_reciprocal_vectors_are_the_half_space_of_the_lattice() -> None:
    triples, k2 = reciprocal_vectors(np.eye(3) * BOX, 2 * np.pi / BOX * 1.5)
    # |n|² = 1 (6 vectors) and 2 (12): half of each
    assert len(triples) == 9 and np.allclose(sorted(set(np.round(k2, 9))), [
        (2 * np.pi / BOX) ** 2, 2 * (2 * np.pi / BOX) ** 2])  # fmt: skip
    for t in triples:
        assert not any(np.array_equal(-t, u) for u in triples)
    skew = np.array([[4.0, 0, 0], [1.0, 3.5, 0], [0.5, 0.3, 3.0]])
    triples, k2 = reciprocal_vectors(skew, 4.0)
    g = 2 * np.pi * np.linalg.inv(skew).T
    assert np.allclose(k2, np.sum((triples @ g) ** 2, axis=1))
    assert np.allclose(skew @ g.T, 2 * np.pi * np.eye(3))


def test_the_modes_equal_the_direct_sum_in_a_changing_triclinic_box() -> None:
    rng = np.random.default_rng(1)
    h = np.array([[4.0, 0, 0], [1.0, 3.5, 0], [0.5, 0.3, 3.0]])
    boxes = np.stack([h, 1.02 * h])
    x = rng.uniform(-5, 5, (2, 7, 3))
    series = density_modes([x], boxes, 4.0)
    triples, _ = reciprocal_vectors(boxes.mean(axis=0), 4.0)
    for f in range(2):
        frac = x[f] @ np.linalg.inv(boxes[f])
        direct = np.exp(2j * np.pi * triples @ frac.T).sum(axis=1)
        assert np.allclose(series.rho[f, 0], direct)
    # an image of a particle changes nothing
    moved = x.copy()
    moved[0, 3] += 2 * boxes[0, 0] - boxes[0, 2]
    assert np.allclose(density_modes([moved], boxes, 4.0).rho, series.rho)


def test_an_ideal_gas_mixture_has_unit_structure_factors() -> None:
    rng = np.random.default_rng(2)
    n_frames = 400
    xs = [rng.uniform(0, BOX, (n_frames, n, 3)) for n in (30, 80)]
    series = density_modes(xs, np.tile(np.eye(3) * BOX, (n_frames, 1, 1)), 5.0)
    k2, modes, sk = partial_sk(series)
    err = 4 / np.sqrt(n_frames * modes)  # ≈ 4σ of an exponentially distributed |ρ|²/N
    for a in range(2):
        assert np.all(np.abs(sk[:, a, a] - 1) < err)
    assert np.all(np.abs(sk[:, 0, 1]) < err)
    limit = s0(k2, modes, sk)
    density = np.array([30, 80]) / BOX**3
    b = limit * np.sqrt(np.outer(density, density))
    # an ideal gas at constant pressure: μ_i = kT ln(p x_i), so d βμ₁/d ln N₁ = x₂
    assert tdf_from_fluctuations(b, density).tdf_molal == pytest.approx(80 / 110, abs=0.05)


def test_one_solvent_gives_gee_2011_eq_10_and_the_volumes_fill_space() -> None:
    rng = np.random.default_rng(3)
    tested = 0
    for _ in range(1000):
        dens = np.array([1.2, 8.0])
        g = rng.normal(0, 0.05, (2, 2))
        g = 0.5 * (g + g.T)
        b = np.diag(dens) + np.outer(dens, dens) * g
        if np.linalg.eigvalsh(b).min() <= 0:  # not a possible fluctuation matrix
            continue
        tested += 1
        got = tdf_from_fluctuations(b, dens)
        assert got.tdf == pytest.approx(1 / (1 + dens[0] * (g[0, 0] - g[0, 1])), rel=1e-12)
        assert float(dens @ got.partial_volumes) == pytest.approx(1.0, rel=1e-12)
    assert tested > 100


def test_a_species_split_by_label_only_changes_nothing() -> None:
    """A solvent whose molecules carry two arbitrary labels: the labels fluctuate ideally."""
    dens = np.array([1.2, 8.0])
    g = np.array([[0.04, -0.03], [-0.03, -0.1]])
    b = np.diag(dens) + np.outer(dens, dens) * g
    part = np.array([0.3, 0.7])
    d3 = np.array([dens[0], *(dens[1] * part)])
    g3 = np.array([[g[0, 0], g[0, 1], g[0, 1]], [g[0, 1], g[1, 1], g[1, 1]],
                   [g[0, 1], g[1, 1], g[1, 1]]])  # fmt: skip
    b3 = np.diag(d3) + np.outer(d3, d3) * g3
    assert tdf_from_fluctuations(b3, d3).tdf == pytest.approx(
        tdf_from_fluctuations(b, dens).tdf, rel=1e-12
    )


def test_lumping_two_different_solvents_is_biased_low() -> None:
    dens = np.array([2.0, 2.9, 7.6])
    v = np.array([0.028, 0.067, 0.1])
    v = v / (dens @ v)
    lump = np.array([[1, 0, 0], [0, 1, 1]])
    for a12, full, lumped in ((-0.2, 1.027, 0.847), (0.2, 0.896, 0.708)):  # note §3
        a = np.diag(1 / dens) + 400 * np.outer(v, v)
        a[0, 1] += a12
        a[1, 0] += a12
        b = np.linalg.inv(a)
        assert tdf_from_fluctuations(b, dens).tdf == pytest.approx(full, abs=5e-4)
        assert tdf_from_fluctuations(lump @ b @ lump.T, lump @ dens).tdf == pytest.approx(
            lumped, abs=5e-4
        )


def _gaussian(s_of_k2, counts, frames: int, seed: int, corr: float = 0.0):
    """Density modes drawn from a prescribed S(k) matrix, with time correlation `corr`."""
    rng = np.random.default_rng(seed)
    triples, k2 = reciprocal_vectors(np.eye(3) * 6.0, 3.2)
    n = len(counts)
    rho = np.empty((frames, n, len(triples)), dtype=complex)
    for j, q in enumerate(k2):
        chol = np.linalg.cholesky(s_of_k2(q) * np.sqrt(np.outer(counts, counts)))
        g = (rng.normal(size=(frames, n)) + 1j * rng.normal(size=(frames, n))) / np.sqrt(2)
        for f in range(1, frames):
            g[f] = corr * g[f - 1] + np.sqrt(1 - corr**2) * g[f]
        rho[:, :, j] = g @ chol.T
    return SkSeries(rho, k2, np.asarray(counts, dtype=float))


H = np.array([[2.0, -0.5, 0.2], [-0.5, 4.0, 1.0], [0.2, 1.0, 5.0]])
L = np.array([[0.05, 0.01, 0.0], [0.01, 0.04, 0.0], [0.0, 0.0, 0.06]])


def test_the_limit_of_a_prescribed_matrix_is_recovered() -> None:
    series = _gaussian(lambda q: np.linalg.inv(H + q * L), [100, 140, 300], 3000, 4)
    k2, modes, sk = partial_sk(series)
    assert np.allclose(s0(k2, modes, sk), np.linalg.inv(H), atol=0.02)
    exact = np.array([np.linalg.inv(H + q * L) for q in k2])
    assert np.allclose(s0(k2, modes, exact), np.linalg.inv(H), atol=1e-12)


def test_the_scalar_form_recovers_an_ornstein_zernike_element() -> None:
    k2 = np.linspace(0.5, 6, 8)
    target = np.array([[0.9, -0.44], [-0.44, 0.6]])
    sk = target[None] / (1 + 0.07 * k2)[:, None, None]
    assert np.allclose(s0(k2, np.ones(8), sk, form="scalar"), target, rtol=1e-10)


def test_the_coulomb_form_projects_the_charge_mode_out() -> None:
    """Wang et al. Eqs. 20–22: exact S(k) with a 1/k² term on the charge direction."""
    x = np.array([0.1, 0.1, 0.8])
    z = np.array([1.0, -1.0, 0.0])
    w = np.sqrt(x) * z
    k2 = np.linspace(0.5, 6, 10)
    kappa2 = 30.0
    unit = np.outer(w, w) / (w @ w)
    sk = np.array([np.linalg.inv(H + q * L + kappa2 / q * unit) for q in k2])
    charge = np.einsum("a,kab,b->k", w, sk, w) / (w @ w)
    assert charge[0] == pytest.approx(k2[0] / kappa2, rel=0.1)  # S_ZZ → k²/κ²
    got = s0(k2, np.ones(10), sk, form="coulomb", charges=z, x=x)
    inv = np.linalg.inv(H)
    exact = inv - inv @ np.outer(w, w) @ inv / (w @ inv @ w)
    assert np.allclose(got, exact, atol=1e-9)
    assert np.allclose(got @ w, 0, atol=1e-9)  # no charge fluctuation is left
    # the neutral matrix form on the same data does not see the projection
    assert not np.allclose(s0(k2, np.ones(10), sk), exact, atol=1e-2)


def test_s0_refusals() -> None:
    k2 = np.linspace(1, 2, 5)
    sk = np.tile(np.eye(2), (5, 1, 1))
    with pytest.raises(ValueError, match="too few"):
        s0(k2[:2], np.ones(2), sk[:2])
    with pytest.raises(ValueError, match="unknown form"):
        s0(k2, np.ones(5), sk, form="cubic")
    with pytest.raises(ValueError, match="charge and number fraction"):
        s0(k2, np.ones(5), sk, form="coulomb")
    with pytest.raises(ValueError, match="positive definite"):
        tdf_from_fluctuations(np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([1.0, 1.0]))


def test_the_relaxation_time_of_the_slowest_mode() -> None:
    series = _gaussian(lambda q: np.linalg.inv(H + q * L), [100, 140, 300], 4000, 5, corr=0.9)
    assert relaxation_time_ps(series, 2.0) == pytest.approx(-2.0 / np.log(0.9), abs=4.0)


N_ION, N_EC, N_EMC, FRAMES = 12, 20, 60, 3000


def _electrolyte(frames: int = FRAMES, seed: int = 6, step: float = 1.0) -> Trajectory:
    """Ideal 'molecules': a fresh random configuration in every frame."""

    def one(q: int, mass: float) -> dict:
        atom = {"name": "X", "element": "Li", "mass_amu": mass, "charge_e": 0.8 * q}
        return {"formal_charge_e": q, "atoms": [atom], "bonds": []}

    doc = {
        "molecule_types": {"Li": one(1, 7), "PF6": one(-1, 145), "EC": one(0, 88),
                           "EMC": one(0, 104)},
        "components": [{"molecule_type": n, "count": k} for n, k in
                       (("Li", N_ION), ("PF6", N_ION), ("EC", N_EC), ("EMC", N_EMC))],
    }  # fmt: skip
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, BOX, (frames, 2 * N_ION + N_EC + N_EMC, 3))
    return Trajectory(system_from_json(doc), np.arange(frames), np.arange(frames) * step, x,
                      np.tile(np.eye(3) * BOX, (frames, 1, 1)), {})  # fmt: skip


def test_the_pipeline_on_an_ideal_electrolyte() -> None:
    """Ideal particles at fixed pressure: V̄ = V/N for every species, TDF_m = 1 − x_ions, TDF = 1."""
    out = thermodynamic_factor(_electrolyte(), "Li", "PF6", k_max_nm=4.0)
    n = 2 * N_ION + N_EC + N_EMC
    t = out["TDF_newman"]
    assert abs(t["mean"] - 1) < 3 * t["sem"] + 0.02
    assert out["TDF_molal"]["mean"] == pytest.approx(1 - 2 * N_ION / n, abs=0.03)
    per_particle = BOX**3 / n * c.NM3_IN_CM3 * c.AVOGADRO_PER_MOL
    assert out["nu_e_cm3_per_mol"]["mean"] == pytest.approx(2 * per_particle, rel=0.05)
    for name in ("EC", "EMC"):
        assert out["partial_volumes_cm3_per_mol"][name]["mean"] == pytest.approx(
            per_particle, rel=0.05
        )
    c_e = N_ION / c.AVOGADRO_PER_MOL / (BOX**3 * c.NM3_IN_CM3) * 1e3
    assert out["_meta"]["c_e_mol_per_L"] == pytest.approx(c_e)
    assert out["structure_factors"]["species"] == ["ions", "EC", "EMC"]


@pytest.mark.parametrize("form", ["matrix", "coulomb"])
def test_fully_bound_ion_pairs_give_one_half(form: str) -> None:
    """An ideal gas of bound pairs has βμ_salt = ln c + const, so ½ dβμ_salt/d ln c = ½. The pairs
    are stretched (σ = 0.15 nm), so the charge structure factor falls as k² and the coulomb form
    has a screening term to fit. Both forms must merge the ions to the same answer."""
    traj = _electrolyte(seed=8)
    x = traj.positions_nm.copy()
    rng = np.random.default_rng(9)
    x[:, N_ION : 2 * N_ION] = x[:, :N_ION] + rng.normal(0, 0.15, (FRAMES, N_ION, 3))
    pairs = Trajectory(traj.system, traj.steps, traj.times_ps, x, traj.boxes_nm, {})
    out = thermodynamic_factor(pairs, "Li", "PF6", k_max_nm=4.0, form=form)
    t = out["TDF_newman"]
    assert abs(t["mean"] - 0.5) < 3 * t["sem"] + 0.02
    n = N_ION + N_EC + N_EMC  # independent units; a pair takes the volume of one of them
    unit = BOX**3 / n * c.NM3_IN_CM3 * c.AVOGADRO_PER_MOL
    assert out["nu_e_cm3_per_mol"]["mean"] == pytest.approx(unit, rel=0.08)


def test_pipeline_refusals() -> None:
    with pytest.raises(ValueError, match="too short"):
        # 20 frames per block against a mode that decorrelates within one frame at best, with a
        # demand of 100 relaxation times
        thermodynamic_factor(_electrolyte(100), "Li", "PF6", k_max_nm=4.0, min_block_per_tau=100)
    with pytest.raises(ValueError, match="1:1"):
        thermodynamic_factor(_electrolyte(100), "Li", "EC", k_max_nm=4.0)
    with pytest.raises(ValueError, match="no reciprocal vector"):
        thermodynamic_factor(_electrolyte(100), "Li", "PF6", k_max_nm=0.5)
    with pytest.raises(ValueError, match="unknown form"):
        thermodynamic_factor(_electrolyte(100), "Li", "PF6", k_max_nm=4.0, form="x")


def test_cli(tmp_path: Path, capsys) -> None:
    import h5py

    traj = _electrolyte(600)
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as fh:
        p = fh.create_group("particles/all/position")
        p["step"], p["time"], p["value"] = traj.steps, traj.times_ps, traj.positions_nm
        e = fh.create_group("particles/all/box/edges")
        e["step"], e["time"], e["value"] = traj.steps, traj.times_ps, traj.boxes_nm
        fh.create_group("parameters/ucmd")["system_json"] = json.dumps(traj.system.doc)
    assert cli.main(["tdf", str(path), "--k-max-per-nm", "4"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["_meta"]["k_max_per_nm"] == 4.0 and out["TDF_newman"]["sem"] > 0
    with pytest.raises(SystemExit):
        cli.main(["tdf", str(path)])
