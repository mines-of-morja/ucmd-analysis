"""Enthalpy of vaporization from a liquid and a gas-phase run (calculation plan A5).

ΔH_vap = ⟨U_gas⟩ − ⟨U_liq⟩/N + RT, per mole of molecules (kJ/mol).

- U_liq is the potential energy of the liquid box with N molecules. It includes the LJ tail
  correction when the run used one.
- U_gas is the potential energy of one molecule alone (a non-periodic run with a thermostat),
  so intramolecular energy cancels to the extent it is the same in both phases.
- RT is p·ΔV for an ideal gas, with the liquid volume neglected.
- No polarization or other correction is applied. A reference that applies one must be compared
  with the correction added on our side, stated with its source.
- Errors are block sems of the two series, combined in quadrature (the runs are independent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import constants as c
from .uncertainty import blocks, mean_sem


def potential_series(path: Path | str, skip_ps: float = 0.0) -> np.ndarray:
    """potential_kjmol of a spec thermo table, from time ≥ skip_ps."""
    with Path(path).open() as fh:
        names = fh.readline().lstrip("# ").strip().split("\t")
        rows = np.loadtxt(fh, ndmin=2)
    t = rows[:, names.index("time_ps")]
    return rows[t >= skip_ps, names.index("potential_kjmol")]


def hvap(u_liquid_kjmol: np.ndarray, n_molecules: int, u_gas_kjmol: np.ndarray,
         temperature_K: float, *, n_blocks: int = 5) -> dict:  # fmt: skip
    liq = [float(np.mean(u_liquid_kjmol[s])) / n_molecules
           for s in blocks(len(u_liquid_kjmol), n_blocks)]  # fmt: skip
    gas = [float(np.mean(u_gas_kjmol[s])) for s in blocks(len(u_gas_kjmol), n_blocks)]
    liq_mean, liq_sem = mean_sem(liq)
    gas_mean, gas_sem = mean_sem(gas)
    rt = c.GAS_CONSTANT_KJ_PER_MOL_K * temperature_K
    return {
        "hvap_kj_per_mol": {
            "mean": gas_mean - liq_mean + rt,
            "sem": float(np.hypot(liq_sem, gas_sem)),
        },
        "u_liquid_per_molecule_kj_per_mol": {"mean": liq_mean, "sem": liq_sem},
        "u_gas_kj_per_mol": {"mean": gas_mean, "sem": gas_sem},
        "RT_kj_per_mol": rt,
        "molecules": n_molecules,
        "temperature_K": temperature_K,
        "liquid_samples": len(u_liquid_kjmol),
        "gas_samples": len(u_gas_kjmol),
        "blocks": n_blocks,
    }
