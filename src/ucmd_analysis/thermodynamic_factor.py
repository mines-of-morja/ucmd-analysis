"""Thermodynamic factor of a 1:1 salt from density fluctuations (A1.7, design/A1.7-tdf.md).

TDF = 1 + d ln f± / d ln c (f± the mean molar activity coefficient; fixed T, p and solvent ratio) in
Newman's normalisation: 1 at infinite dilution, half of Lehnert 2025's value.

The fluctuation matrix B_αβ = √(c_α c_β) S⁰_αβ of the species {ions together, solvent 1, …}
gives, with A = B⁻¹, u = A c and w = c·u (Eq. 1 of the note):

    TDF_m = c₁ (A₁₁ − u₁²/w),   V̄_α = u_α/w,   TDF = TDF_m / (1 − c₁ V̄₁)

The solvents are never lumped: that would let their ratio relax around the ions, which is another
derivative and biased low. For one solvent the formula is Gee 2011, Eq. 10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as c
from .io import Trajectory
from .molecules import centres_of_mass
from .structure_factor import SkSeries, density_modes, partial_sk, s0
from .uncertainty import blocks, mean_sem


@dataclass(frozen=True)
class TdfValue:
    tdf_molal: float
    tdf: float  # molar, Newman
    partial_volumes: np.ndarray  # per particle, in the inverse unit of the densities


def tdf_from_fluctuations(b: np.ndarray, density: np.ndarray) -> TdfValue:
    """Equation 1; species 0 holds all ions, `b` = ⟨δN_α δN_β⟩/V at k → 0."""
    if not np.linalg.eigvalsh(0.5 * (b + b.T)).min() > 0:
        raise ValueError(
            "the fluctuation matrix is not positive definite: the run is too short or the "
            "k range does not suit the fit"
        )
    a = np.linalg.inv(b)
    u = a @ density
    w = float(density @ u)
    molal = float(density[0] * (a[0, 0] - u[0] ** 2 / w))
    v = u / w
    return TdfValue(molal, molal / (1 - float(density[0] * v[0])), v)


def relaxation_time_ps(series: SkSeries, dt_ps: float, species: int = 0) -> float:
    """1/e time of ρ(k, t) of one species at the smallest |k| of the box."""
    first = series.k2_nm2 <= series.k2_nm2[0] * (1 + 1e-6)
    r = series.rho[:, species, first]
    n = len(r)
    f = np.fft.fft(r, n=2 * n, axis=0)
    acf = np.fft.ifft(np.abs(f) ** 2, axis=0)[:n].real.sum(axis=1) / (n - np.arange(n))
    below = np.flatnonzero(acf < acf[0] / np.e)
    return float("inf") if len(below) == 0 else float(below[0] * dt_ps)


def thermodynamic_factor(traj: Trajectory, cation: str, anion: str, *, k_max_nm: float,
                         n_blocks: int = 5, form: str = "matrix",
                         min_block_per_tau: float = 5.0) -> dict:  # fmt: skip
    """TDF with block errors. `form`: `matrix` or `scalar` (ions as one species), or `coulomb`
    (cation and anion apart in the fit, merged afterwards). Blocks shorter than
    `min_block_per_tau` relaxation times of the slowest mode are refused."""
    sp = traj.system.species
    if not (sp[cation].count == sp[anion].count
            and sp[cation].formal_charge == -sp[anion].formal_charge != 0):  # fmt: skip
        raise ValueError("the TDF here is for a 1:1 salt with equal numbers of both ions")
    solvents = [n for n in sp if n not in (cation, anion)]
    charged = [n for n in solvents if sp[n].formal_charge != 0 or abs(sp[n].charge) > 1e-9]
    if not solvents or charged:
        raise ValueError(f"the TDF needs neutral solvents; found {charged or 'no solvent'}")
    if form not in ("matrix", "scalar", "coulomb"):
        raise ValueError(f"unknown form {form!r}: scalar, matrix or coulomb")
    com = {n: centres_of_mass(traj, sp[n]) for n in sp}
    ions = np.concatenate([com[cation], com[anion]], axis=1)
    solvent_x = [com[n] for n in solvents]
    apart = form == "coulomb"
    groups = [com[cation], com[anion], *solvent_x] if apart else [ions, *solvent_x]
    series = density_modes(groups, traj.boxes_nm, k_max_nm)
    dt = float(np.mean(np.diff(traj.times_ps)))
    tau = max(relaxation_time_ps(series, dt, s) for s in range(len(groups)))
    block_ps = (len(traj) // n_blocks) * dt
    if not block_ps >= min_block_per_tau * tau:
        raise ValueError(
            f"blocks of {block_ps:.4g} ps are too short: the slowest mode at the smallest k "
            f"relaxes in {tau:.4g} ps and a block needs {min_block_per_tau:g} times that"
        )
    volume = traj.volumes_nm3()
    counts = np.array([g.shape[1] for g in groups], dtype=float)
    z = np.array([1.0, -1.0] + [0.0] * len(solvents)) if apart else None
    merge = np.eye(len(groups))
    if apart:
        merge = np.vstack([merge[0] + merge[1], merge[2:]])

    def value(frames: slice) -> TdfValue:
        k2, modes, sk = partial_sk(series, frames)
        limit = s0(k2, modes, sk, form=form, charges=z, x=counts / counts.sum())
        density = counts / float(volume[frames].mean())  # nm⁻³
        b = limit * np.sqrt(np.outer(density, density))
        return tdf_from_fluctuations(merge @ b @ merge.T, merge @ density)

    per = [value(s) for s in blocks(len(traj), n_blocks)]
    whole = value(slice(None))
    molar = c.NM3_IN_CM3 * c.AVOGADRO_PER_MOL  # nm³ per particle -> cm³/mol

    def stat(values) -> dict[str, float]:
        return dict(zip(("mean", "sem"), mean_sem(values), strict=True))

    k2, modes, sk = partial_sk(series)
    labels = [cation, anion, *solvents] if apart else ["ions", *solvents]
    return {
        "TDF_newman": {**stat([p.tdf for p in per]), "whole_run": whole.tdf},
        "TDF_molal": {**stat([p.tdf_molal for p in per]), "whole_run": whole.tdf_molal},
        # the salt holds two ions; V̄ of one ion of the merged species is half of it
        "nu_e_cm3_per_mol": stat([2 * p.partial_volumes[0] * molar for p in per]),
        "partial_volumes_cm3_per_mol": {
            n: stat([p.partial_volumes[i + 1] * molar for p in per])
            for i, n in enumerate(solvents)
        },
        "structure_factors": {
            "species": labels, "k_per_nm": np.sqrt(k2).tolist(), "modes": modes.tolist(),
            "S": sk.tolist(),
        },
        "_meta": {
            "form": form, "k_max_per_nm": k_max_nm, "k_min_per_nm": float(np.sqrt(k2[0])),
            "shells": len(k2), "frames": len(traj), "frame_interval_ps": dt, "blocks": n_blocks,
            "block_ps": block_ps, "slowest_mode_tau_ps": tau,
            "c_e_mol_per_L": sp[cation].count / float(volume.mean()) / molar * 1e3,
            "normalisation": "Newman: 1 at infinite dilution (Lehnert 2025 is twice this)",
        },
    }  # fmt: skip
