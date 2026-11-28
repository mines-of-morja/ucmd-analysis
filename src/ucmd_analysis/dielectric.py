"""Static dielectric constant from the fluctuation of the total dipole (calculation plan A7).

- **Dipole:** M(t) = Σ over the selected molecules of Σᵢ qᵢ rᵢ, in e·nm, with the model's own
  (possibly scaled) charges.
  - Only neutral molecule types count. Then each molecular dipole is independent of the origin,
    and whole molecules in any periodic image give the same M (positions are unwrapped,
    molecules whole).
  - Ions are left out. Their contribution grows with time (it is the current, a part of the
    conductivity), so it has no static fluctuation to average.
- **Formula:** Ewald and PME sum with conducting ("tin-foil") boundaries, so
  ε = 1 + (⟨M²⟩ − ⟨M⟩²)/(3 ε₀ ⟨V⟩ k_B T)
  (Neumann, Mol. Phys. 50, 841 (1983)). ⟨V⟩ is the mean volume; under NPT this neglects the
  correlation of M² with V.
- **Convergence is slow:** water needs several ns. The block spread shows whether the series is
  long enough.
"""

from __future__ import annotations

import numpy as np

from . import constants as c
from .io import Trajectory
from .uncertainty import blocks as block_slices
from .uncertainty import mean_sem


def neutral_types(traj: Trajectory) -> list[str]:
    return [n for n, m in traj.system.species.items() if abs(m.charge) < 1e-6]


def total_dipole(traj: Trajectory, molecule_types: list[str]) -> np.ndarray:
    """(frames, 3) total dipole of the selected molecule types, e·nm."""
    total = np.zeros((len(traj), 3))
    for name in molecule_types:
        mol = traj.system.species[name]
        if abs(mol.charge) > 1e-6:
            raise ValueError(f"{name} is not neutral (charge {mol.charge:+.4f} e)")
        q = traj.system.charges[mol.atoms[0]]  # (atoms per molecule,)
        x = traj.positions_nm[:, mol.atoms]  # (frames, molecules, atoms, 3)
        # relative to the molecule's first atom: exact for a neutral molecule, better rounded
        mu = np.einsum("k,fmkd->fd", q, x - x[:, :, :1])
        total += mu
    return total


def epsilon(m: np.ndarray, volume_nm3: float, temperature_K: float) -> float:
    var = float(np.mean((m - m.mean(axis=0)) ** 2) * 3.0)  # ⟨M²⟩ − ⟨M⟩²
    kt = c.BOLTZMANN_KJMOL_PER_K * temperature_K
    return 1.0 + var / (3.0 * c.EPSILON0_E2_PER_KJMOL_NM * volume_nm3 * kt)


def dielectric(traj: Trajectory, temperature_K: float, *, molecule_types: list[str] | None = None,
               blocks: int = 5) -> dict:  # fmt: skip
    """ε over the whole series and as block mean ± sem, with the inputs."""
    types = molecule_types or neutral_types(traj)
    m = total_dipole(traj, types)
    v = float(np.mean(traj.volumes_nm3()))
    per = [epsilon(m[s], float(np.mean(traj.volumes_nm3()[s])), temperature_K)
           for s in block_slices(len(m), blocks)]  # fmt: skip
    mean, sem = mean_sem(per)
    return {
        "epsilon": epsilon(m, v, temperature_K),
        "epsilon_blocks": {"mean": mean, "sem": sem, "blocks": blocks, "values": per},
        "mean_dipole_e_nm": m.mean(axis=0).tolist(),
        "dipole_variance_e2_nm2": float(np.mean((m - m.mean(axis=0)) ** 2) * 3.0),
        "volume_nm3": v,
        "temperature_K": temperature_K,
        "frames": len(m),
        "molecule_types": types,
    }
