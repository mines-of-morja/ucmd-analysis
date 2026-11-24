"""Reference frames of the transference number (A1.5; ADR 0013 and design/A1.5.md of UnitcellMD).

t₊ depends on the reference velocity; κ does not. Three frames are reported for a 1:1 salt:

- COM: barycentric, the native MD frame;
- SOL: the mass-weighted velocity of all neutral species together (EC + EMC as one solvent),
  Newman's t₊⁰;
- VOL: the volume-averaged velocity (eNMR, Lehnert 2025), from SOL and the partial molar volumes:
  t₊^VOL = (1 − c_e ν_e) t₊^SOL + c_e (ν_e − ν₊)                       (Eq. 3 of the note)

A frame with reference displacement R changes the collective displacement of ion species i to
X_i = Σ_a Δr_a − N_i R. SOL is computed from such displacements; Eq. 2 of the note,
t₊^SOL = (t₊^COM − ω₋)/ω₀ with the mass fractions ω, is an independent check of it.

ν_e (the salt) is an input, in cm³/mol, whose source the caller records. How it splits into ν₊ and
ν₋ is a convention; both conventions of ADR 0013 are always reported, A as the central value and B
as its band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import FitWindow, slope
from .io import System
from .msd import cross_msd, msd_series


@dataclass(frozen=True)
class IonVolumeConvention:
    nu_plus_cm3_per_mol: float
    err_cm3_per_mol: float | None  # as stated by the source; None if it states none
    source: str


#: ADR 0013: A is the central value, B the stated band.
ION_VOLUME_CONVENTIONS = {
    "A": IonVolumeConvention(
        -9.0, 2.0, "Li+ in PC at 25 C, TATB/TPTB scale: Marcus & Hefter, Chem. Rev. 104, 3405 "
        "(2004), Table 15",
    ),
    "B": IonVolumeConvention(
        1.1, None, "intrinsic volume of Li+: Marcus & Hefter, Chem. Rev. 104, 3405 (2004), Table 40"
    ),
}  # fmt: skip


@dataclass(frozen=True)
class FrameValue:
    t_plus: float
    t_minus: float
    total: float  # L₊₊ + L₋₋ − 2L₊₋ (∝ κ), slope units nm²/ps


def frame_t_plus(x_plus: np.ndarray, x_minus: np.ndarray, lags_ps: np.ndarray,
                 window: FitWindow) -> FrameValue:  # fmt: skip
    """t₊ of a 1:1 salt from the collective ion displacements X_i (frames, 3) in one frame."""
    lpp = slope(lags_ps, msd_series(x_plus), window)
    lmm = slope(lags_ps, msd_series(x_minus), window)
    lpm = slope(lags_ps, cross_msd(x_plus, x_minus), window)
    total = lpp + lmm - 2 * lpm
    return FrameValue((lpp - lpm) / total, (lmm - lpm) / total, total)


def sol_from_com(t_plus_com: float, omega_minus: float, omega_0: float) -> float:
    """Eq. 2 of the note; ω₋ and ω₀ are the mass fractions of the anion and of the solvent."""
    return (t_plus_com - omega_minus) / omega_0


def vol_from_sol(t_plus_sol: float, c_e_mol_per_cm3: float, nu_e_cm3_per_mol: float,
                 nu_plus_cm3_per_mol: float) -> float:  # fmt: skip
    """Eq. 3 of the note, with c₀ν₀ = 1 − c_e ν_e and ν₋ = ν_e − ν₊."""
    c_e, nu_e = c_e_mol_per_cm3, nu_e_cm3_per_mol
    return (1 - c_e * nu_e) * t_plus_sol + c_e * (nu_e - nu_plus_cm3_per_mol)


@dataclass(frozen=True)
class SolventFrame:
    """The solvent of a 1:1 electrolyte: all neutral species, as one mass-weighted body."""

    names: tuple[str, ...]
    masses_amu: dict[str, float]  # total mass per species
    omega_minus: float
    omega_0: float

    def displacement(self, com: dict[str, np.ndarray]) -> np.ndarray:
        """Centre of mass of the solvent, (frames, 3), in the frame of the given coordinates."""
        total = sum(self.masses_amu[n] for n in self.names)
        return sum(self.masses_amu[n] / total * com[n].mean(axis=1) for n in self.names)


def solvent_frame(system: System, cation: str, anion: str) -> SolventFrame | str:
    """The solvent frame, or the reason (a sentence) why there is none."""
    sp = system.species
    if not (sp[cation].count == sp[anion].count
            and sp[cation].formal_charge == -sp[anion].formal_charge):  # fmt: skip
        return "the SOL and VOL frames need a 1:1 salt with equal numbers of cations and anions"
    names = tuple(n for n in sp if n not in (cation, anion))
    if not names:
        return "no solvent: every species is an ion"
    for n in names:
        if sp[n].formal_charge != 0 or abs(sp[n].charge) > 1e-9:
            return f"{n} is charged ({sp[n].formal_charge:+d} e), so it is not a solvent"
    masses = {n: float(sp[n].masses.sum()) * sp[n].count for n in sp}
    total = sum(masses.values())
    return SolventFrame(names, masses, masses[anion] / total, sum(masses[n] for n in names) / total)
