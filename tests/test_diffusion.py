"""Self-diffusion of neutral species (pure solvents such as water) and the refusal of neutral
ions in `transport`.

Known answer: a random walker with Gaussian steps of standard deviation s per axis and frame
(dt apart) has D = s²/(2 dt). Pure water has no cation and anion, so `ucmd-analyze transport`
had nothing to compute and stopped with a ZeroDivisionError in the ionicity (2026-11-21).
"""

from __future__ import annotations

import numpy as np
import pytest

from ucmd_analysis import constants as c
from ucmd_analysis.io import Trajectory, system_from_json
from ucmd_analysis.transport import diffusion, transport

N, FRAMES, DT_PS, STEP_NM = 100, 4000, 1.0, 0.05
D_TRUE_NM2_PS = STEP_NM**2 / (2 * DT_PS)  # 1.25e-3 nm²/ps


def _trajectory(box_nm=(5.0, 5.0, 5.0), seed: int = 1) -> Trajectory:
    rng = np.random.default_rng(seed)
    doc = {
        "molecule_types": {"W": {"formal_charge_e": 0, "bonds": [],
                                 "atoms": [{"element": "O", "mass_amu": 18.0, "charge_e": 0.0}]}},
        "components": [{"molecule_type": "W", "count": N}],
    }  # fmt: skip
    x = np.cumsum(rng.normal(0.0, STEP_NM, (FRAMES, N, 3)), axis=0)
    return Trajectory(
        system_from_json(doc), np.arange(FRAMES), np.arange(FRAMES) * DT_PS, x,
        np.tile(np.diag(box_nm), (FRAMES, 1, 1)), {},
    )  # fmt: skip


def test_random_walkers_give_their_diffusion_coefficient() -> None:
    out = diffusion(_trajectory(), 300.0)
    d = out["D_W_m2_per_s"]
    expected = D_TRUE_NM2_PS * c.NM2_PER_PS_IN_M2_PER_S * (1 - 1 / N)  # barycentric frame
    assert d["mean"] == pytest.approx(expected, abs=max(3 * d["sem"], 0.02 * expected))
    assert out["_meta"]["frames"] == FRAMES and "D_W_YH_m2_per_s" not in out


def test_the_finite_size_correction_is_added_for_a_cubic_box() -> None:
    plain = diffusion(_trajectory(), 300.0)
    eta = 0.9e-3
    out = diffusion(_trajectory(), 300.0, viscosity_Pa_s=eta)
    shift = c.BOLTZMANN_J_PER_K * 300.0 * c.YEH_HUMMER_XI_CUBIC / (6 * np.pi * eta * 5.0e-9)
    assert out["D_W_YH_m2_per_s"]["mean"] == pytest.approx(
        plain["D_W_m2_per_s"]["mean"] + shift, rel=1e-12
    )
    with pytest.raises(ValueError, match="cubic"):
        diffusion(_trajectory(box_nm=(5.0, 5.0, 4.0)), 300.0, viscosity_Pa_s=eta)
    with pytest.raises(ValueError, match="positive"):
        diffusion(_trajectory(), 300.0, viscosity_Pa_s=0.0)


def test_species_can_be_selected_and_unknown_ones_are_refused() -> None:
    assert "D_W_m2_per_s" in diffusion(_trajectory(), 300.0, species=["W"])
    with pytest.raises(ValueError, match="no species 'Q'"):
        diffusion(_trajectory(), 300.0, species=["Q"])


def test_transport_refuses_neutral_ions_and_points_to_diffusion() -> None:
    with pytest.raises(ValueError, match=r"neutral.*diffusion"):
        transport(_trajectory(), 300.0, "W", "W")
