"""Package of the binary mixture of Cheng 2022 (README.md): 2000 A + 2000 B on an fcc lattice.

    python make_system.py OUT.ucsys

Reduced units of Cheng: sigma = 1 and epsilon = 1 (the B-B pair), epsilon_AA = 1.2,
epsilon_AB = 1.1 (arithmetic mean, so a spec 0.1.2 pair override), equal masses. Rahman's argon
(1964) fixes the scale; the structure factor in reduced units does not depend on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ucmd_analysis import constants as c

CELLS = 10  # 4 * 10**3 = 4000 sites
RHO_START = 0.6  # sites per sigma^3; a start only, the barostat finds the density
SIGMA_NM = 0.34  # Rahman 1964
EPSILON_OVER_KB_K = 120.0  # Rahman 1964
MASS_AMU = 39.948
SEED = 2022

SCALE = {
    "sigma_nm": SIGMA_NM,
    "epsilon_kjmol": EPSILON_OVER_KB_K * c.BOLTZMANN_KJMOL_PER_K,
    "mass_amu": MASS_AMU,
}
TAU_PS = SIGMA_NM * (MASS_AMU / SCALE["epsilon_kjmol"]) ** 0.5  # sqrt(amu nm2/(kJ/mol)) = ps
BAR = c.KJMOL_IN_J / c.NM_IN_M**3 / c.BAR_IN_PA  # 1 kJ/mol per nm3, in bar
PRESSURE_UNIT_BAR = SCALE["epsilon_kjmol"] / SIGMA_NM**3 * BAR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out
    eps = SCALE["epsilon_kjmol"]
    box = (4 * CELLS**3 / RHO_START) ** (1 / 3) * SIGMA_NM
    a = box / CELLS
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    grid = np.array(
        [[i, j, k] for i in range(CELLS) for j in range(CELLS) for k in range(CELLS)], dtype=float
    )
    sites = ((grid[:, None, :] + basis[None]).reshape(-1, 3)) * a
    sites = sites[np.random.default_rng(SEED).permutation(len(sites))]  # A: first half, B: second

    def atom(name: str) -> dict:
        one = {"name": name, "element": "Ar", "mass_amu": MASS_AMU, "charge_e": 0.0, "type": name}
        return {"formal_charge_e": 0, "atoms": [one], "bonds": []}

    doc = {
        "format": "ucmd-system",
        "spec_version": "0.1.2",
        "name": "cheng2022-binary-4000",
        "description": "Cheng 2022 binary repulsive LJ mixture, x_A = 0.5, fcc start.",
        "units": {
            "length": "nm",
            "time": "ps",
            "mass": "amu",
            "energy": "kJ/mol",
            "charge": "e",
            "temperature": "K",
            "angle": "deg",
        },
        "periodicity": [True, True, True],
        "cell_nm": [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
        "molecule_types": {"A": atom("A"), "B": atom("B")},
        "components": [
            {"molecule_type": "A", "count": 2000},
            {"molecule_type": "B", "count": 2000},
        ],
        "atom_count": 4 * CELLS**3,
        "forcefield": {
            "name": "Binary Lennard-Jones, Cheng 2022",
            "combination_rule": "lorentz-berthelot",
            "scale_14_lj": 1.0,
            "scale_14_coulomb": 1.0,
            "atom_types": {
                "A": {"sigma_nm": SIGMA_NM, "epsilon_kjmol": 1.2 * eps},
                "B": {"sigma_nm": SIGMA_NM, "epsilon_kjmol": 1.0 * eps},
            },
            "pair_overrides": [
                {
                    "types": ["A", "B"],
                    "sigma_nm": SIGMA_NM,
                    "epsilon_kjmol": 1.1 * eps,
                    "description": "epsilon_AB = (epsilon_AA + epsilon_BB)/2, Cheng 2022",
                }
            ],
        },
        "provenance": {
            "created_by": "ucmd-analysis/validation/a17-lj/make_system.py",
            "created_utc": "2026-11-25T00:00:00Z",
            "sources": [
                {
                    "what": "mixture in reduced units, state point",
                    "citation": "B. Cheng, arXiv:2205.03070v2; J. Chem. Phys. 157, 121101 (2022)",
                },
                {"what": "argon scale", "citation": "A. Rahman, Phys. Rev. 136, A405 (1964)"},
            ],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "positions.npy", sites)
    (out / "system.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"tau = {TAU_PS:.6f} ps, dt 1e-3 tau = {TAU_PS:.6f} fs, "
        f"T 1.2 = {1.2 * EPSILON_OVER_KB_K:.3f} K, P 2 = {2 * PRESSURE_UNIT_BAR:.3f} bar, "
        f"2^(1/6) sigma = {2 ** (1 / 6) * SIGMA_NM:.8f} nm, box {box:.4f} nm"
    )


if __name__ == "__main__":
    main()
