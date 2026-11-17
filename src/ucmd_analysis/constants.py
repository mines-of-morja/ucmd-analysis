"""Physical constants for the analysis (CODATA 2018, exact SI values; ADR 0001).

The only module with numeric constants. Trajectories are in spec units (nm, ps, amu, e, K);
transport results are reported in SI (m²/s, S/m) and common lab units (cm²/s, mS/cm).
"""

from typing import Final

AVOGADRO_PER_MOL: Final = 6.02214076e23
BOLTZMANN_J_PER_K: Final = 1.380649e-23
ELEMENTARY_CHARGE_C: Final = 1.602176634e-19

NM_IN_M: Final = 1e-9
PS_IN_S: Final = 1e-12
NM2_PER_PS_IN_M2_PER_S: Final = NM_IN_M**2 / PS_IN_S  # 1 nm²/ps = 1e-6 m²/s
M2_PER_S_IN_CM2_PER_S: Final = 1e4
S_PER_M_IN_MS_PER_CM: Final = 10.0  # 1 S/m = 10 mS/cm
AMU_IN_G: Final = 1.0 / AVOGADRO_PER_MOL  # 1 amu = 1 g/mol / N_A
NM3_IN_CM3: Final = 1e-21
