"""Molecular centres of mass and the barycentric frame (PLAN.md A1.2).

Positions are unwrapped and molecules whole (spec units.md §3), so a centre of mass is a plain
mass-weighted mean. The barycentric frame subtracts the system's centre of mass in every frame;
Onsager coefficients are defined there (Fong et al. 2020; Lehnert 2025 Eq. 3).
"""

from __future__ import annotations

import numpy as np

from .io import Molecules, Trajectory


def centres_of_mass(traj: Trajectory, mol: Molecules) -> np.ndarray:
    """(frames, molecules, 3) in nm."""
    x = traj.positions_nm[:, mol.atoms, :]  # frames, molecules, atoms, 3
    w = mol.masses / mol.masses.sum()
    return np.einsum("fmad,a->fmd", x, w)


def system_centre_of_mass(traj: Trajectory) -> np.ndarray:
    """(frames, 3) in nm."""
    m = traj.system.masses
    return np.einsum("fad,a->fd", traj.positions_nm, m / m.sum())


def barycentric(traj: Trajectory, mol: Molecules) -> np.ndarray:
    """Centres of mass relative to the system's centre of mass, (frames, molecules, 3) in nm."""
    return centres_of_mass(traj, mol) - system_centre_of_mass(traj)[:, None, :]
