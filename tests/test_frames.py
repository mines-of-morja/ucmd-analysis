"""t₊ in the COM, SOL and VOL frames (A1.5; ADR 0013; design/A1.5.md of UnitcellMD).

Known answers: the frame velocities are fixed linear combinations of the species' velocities, so
t₊ of one frame follows exactly from t₊ of another (Eqs. 2 and 3 of the design note). A published
conversion is reproduced as well: Lehnert 2025's COM fit against their SOL fit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ucmd_analysis import cli
from ucmd_analysis import constants as c
from ucmd_analysis.frames import (
    ION_VOLUME_CONVENTIONS,
    frame_t_plus,
    sol_from_com,
    vol_from_sol,
)
from ucmd_analysis.io import Trajectory, system_from_json
from ucmd_analysis.molecules import barycentric
from ucmd_analysis.transport import FitWindow, transport

N_IONS, N_EC, N_EMC, FRAMES, DT_PS, BOX_NM = 20, 40, 60, 2000, 0.5, 4.0
MASS = {"Li": 6.94, "PF6": 144.964180970, "EC": 88.062, "EMC": 104.105}
NU_E, NU_E_ERR = 55.7, 3.0  # cm³/mol; the test needs only some plausible numbers


def _doc(n_anions: int = N_IONS, solvent_charge: int = 0) -> dict:
    def one(element: str, name: str, q: int) -> dict:
        atom = {"name": "X", "element": element, "mass_amu": MASS[name], "charge_e": 0.75 * q}
        return {"formal_charge_e": q, "atoms": [atom], "bonds": []}

    return {
        "molecule_types": {
            "Li": one("Li", "Li", 1), "PF6": one("P", "PF6", -1),
            "EC": one("C", "EC", solvent_charge), "EMC": one("C", "EMC", 0),
        },
        "components": [
            {"molecule_type": "Li", "count": N_IONS},
            {"molecule_type": "PF6", "count": n_anions},
            {"molecule_type": "EC", "count": N_EC},
            {"molecule_type": "EMC", "count": N_EMC},
        ],
    }  # fmt: skip


def _trajectory(seed: int = 5, **kw) -> Trajectory:
    """Random walkers; every ion pair also shares part of its motion with one EC molecule, so the
    frames differ from each other by more than the round-off."""
    doc = _doc(**kw)
    rng = np.random.default_rng(seed)
    n_anions = doc["components"][1]["count"]
    sigma = [0.03] * N_IONS + [0.02] * n_anions + [0.04] * N_EC + [0.05] * N_EMC
    steps = rng.normal(size=(FRAMES, len(sigma), 3)) * np.array(sigma)[None, :, None]
    shared = rng.normal(0.0, 0.02, (FRAMES, N_IONS, 3))
    steps[:, :N_IONS] += shared
    steps[:, N_IONS + n_anions : N_IONS + n_anions + N_IONS] += shared
    x = np.cumsum(steps, axis=0)
    return Trajectory(system_from_json(doc), np.arange(FRAMES), np.arange(FRAMES) * DT_PS, x,
                      np.tile(np.eye(3) * BOX_NM, (FRAMES, 1, 1)), {})  # fmt: skip


def test_sol_from_displacements_equals_eq2_in_every_block() -> None:
    """Acceptance (ADR 0013, 5): t₊^SOL = (t₊^COM − ω₋)/ω₀ to 1e-9, block by block."""
    traj = _trajectory()
    out = transport(traj, 300.0, "Li", "PF6", keep_blocks=True)
    m = {n: MASS[n] * traj.system.species[n].count for n in MASS}
    total = sum(m.values())
    w_minus, w_0 = m["PF6"] / total, (m["EC"] + m["EMC"]) / total
    for com, sol in zip(out["_blocks"]["t_plus_COM"], out["_blocks"]["t_plus_SOL"], strict=True):
        assert sol == pytest.approx(sol_from_com(com, w_minus, w_0), abs=1e-9)
        assert abs(sol - com) > 1e-2  # the frames really differ here
    meta = out["_meta"]["frames"]
    assert meta["omega_minus"] == pytest.approx(w_minus, rel=1e-12)
    assert meta["omega_0"] == pytest.approx(w_0, rel=1e-12)
    assert meta["max_abs_SOL_minus_Eq2"] < 1e-9


def test_vol_from_displacements_equals_eq3_and_kappa_is_frame_independent() -> None:
    """The volume-average reference displacement, summed by hand, gives Eq. 3; t₊ + t₋ = 1 and the
    charge transport |X₊ − X₋|² are the same in every frame."""
    traj = _trajectory()
    sp = traj.system.species
    com = {n: barycentric(traj, sp[n]) for n in sp}
    mean = {n: com[n].mean(axis=1) for n in com}
    m_ec, m_emc = MASS["EC"] * N_EC, MASS["EMC"] * N_EMC
    sol = (m_ec * mean["EC"] + m_emc * mean["EMC"]) / (m_ec + m_emc)
    lags = np.arange(FRAMES) * DT_PS
    s_p, s_m = com["Li"].sum(axis=1), com["PF6"].sum(axis=1)
    c_e = N_IONS / c.AVOGADRO_PER_MOL / (BOX_NM**3 * c.NM3_IN_CM3)  # mol/cm³
    nu_p = ION_VOLUME_CONVENTIONS["A"].nu_plus_cm3_per_mol
    vol = (1 - c_e * NU_E) * sol + c_e * nu_p * mean["Li"] + c_e * (NU_E - nu_p) * mean["PF6"]
    w = FitWindow()
    results = {}
    for name, r in (("COM", 0 * sol), ("SOL", sol), ("VOL", vol), ("EC", mean["EC"])):
        results[name] = frame_t_plus(s_p - N_IONS * r, s_m - N_IONS * r, lags, w)
        assert results[name].t_plus + results[name].t_minus == pytest.approx(1.0, abs=1e-12)
        assert results[name].total == pytest.approx(results["COM"].total, rel=1e-9)
    direct = results["VOL"].t_plus
    assert direct == pytest.approx(
        vol_from_sol(results["SOL"].t_plus, c_e, NU_E, nu_p), abs=1e-9
    )  # fmt: skip
    # the rule is general: t₊ relative to the cation is 0, relative to the anion 1
    for ref, expected in (("Li", 0.0), ("PF6", 1.0)):
        r = mean[ref]
        got = frame_t_plus(s_p - N_IONS * r, s_m - N_IONS * r, lags, w).t_plus
        assert got == pytest.approx(expected, abs=1e-9)


def test_one_prescribed_solvent_drift_has_the_exact_answer() -> None:
    """Deterministic motion: every species moves at a constant velocity, so t₊ of the frame with
    reference v is (v₊ − v)/(v₊ − v₋) (z = ±1), by hand."""
    doc = _doc()
    v = {"Li": 2.0, "PF6": -1.0, "EC": 0.5, "EMC": 0.1}  # along x, nm/ps
    per_atom = np.repeat([v[k["molecule_type"]] for k in doc["components"]],
                         [k["count"] for k in doc["components"]])  # fmt: skip
    t = np.arange(400) * DT_PS
    x = np.zeros((len(t), len(per_atom), 3))
    x[:, :, 0] = t[:, None] * per_atom
    traj = Trajectory(system_from_json(doc), np.arange(len(t)), t, x,
                      np.tile(np.eye(3) * BOX_NM, (len(t), 1, 1)), {})  # fmt: skip
    out = transport(traj, 300.0, "Li", "PF6", nu_e_cm3_per_mol=NU_E)
    m = {n: MASS[n] * traj.system.species[n].count for n in MASS}
    v_sol = (m["EC"] * v["EC"] + m["EMC"] * v["EMC"]) / (m["EC"] + m["EMC"])
    v_com = sum(m[n] * v[n] for n in m) / sum(m.values())

    def hand(ref: float) -> float:
        return (v["Li"] - ref) / (v["Li"] - v["PF6"])

    assert out["t_plus_COM"]["mean"] == pytest.approx(hand(v_com), abs=1e-9)
    assert out["t_plus_SOL"]["mean"] == pytest.approx(hand(v_sol), abs=1e-9)
    c_e = N_IONS / c.AVOGADRO_PER_MOL / (BOX_NM**3 * c.NM3_IN_CM3)
    for name, conv in ION_VOLUME_CONVENTIONS.items():
        nu_p = conv.nu_plus_cm3_per_mol
        v_vol = ((1 - c_e * NU_E) * v_sol + c_e * nu_p * v["Li"]
                 + c_e * (NU_E - nu_p) * v["PF6"])  # fmt: skip
        assert out[f"t_plus_VOL_{name}"]["mean"] == pytest.approx(hand(v_vol), abs=1e-9)


def test_lehnerts_com_fit_gives_their_sol_fit() -> None:
    """Published conversion: Lehnert 2025 SI Table S5 (COM: 0.364 + 0.046 c) against Table S6
    (SOL: 0.362 − 0.031 c) at 1 M, with their box density 1.202 g/cm³ (ledger). Printed to three
    decimals; 0.0015 covers the rounding of the four printed coefficients."""
    rho, c_e = 1202.0, 1.0  # g/L, mol/L
    w_minus, w_plus = MASS["PF6"] * c_e / rho, MASS["Li"] * c_e / rho
    got = sol_from_com(0.364 + 0.046 * c_e, w_minus, 1 - w_minus - w_plus)
    assert got == pytest.approx(0.362 - 0.031 * c_e, abs=0.0015)


def test_the_conventions_are_those_of_adr_0013() -> None:
    a, b = ION_VOLUME_CONVENTIONS["A"], ION_VOLUME_CONVENTIONS["B"]
    assert (a.nu_plus_cm3_per_mol, a.err_cm3_per_mol) == (-9.0, 2.0)
    assert (b.nu_plus_cm3_per_mol, b.err_cm3_per_mol) == (1.1, None)
    assert "Marcus" in a.source and "Marcus" in b.source


def test_vol_needs_nu_e_and_reports_its_error_parts() -> None:
    traj = _trajectory()
    out = transport(traj, 300.0, "Li", "PF6")
    assert "t_plus_SOL" in out
    assert not [k for k in out if k.startswith("t_plus_VOL")]
    out = transport(traj, 300.0, "Li", "PF6",
                    nu_e_cm3_per_mol=NU_E, nu_e_err_cm3_per_mol=NU_E_ERR)  # fmt: skip
    c_e = N_IONS / c.AVOGADRO_PER_MOL / (BOX_NM**3 * c.NM3_IN_CM3)
    sol = out["t_plus_SOL"]["mean"]
    a, b = out["t_plus_VOL_A"], out["t_plus_VOL_B"]
    assert a["mean"] - b["mean"] == pytest.approx(c_e * 10.1, rel=1e-9)  # ν₊: 1.1 − (−9)
    assert a["sem"] == pytest.approx((1 - c_e * NU_E) * out["t_plus_SOL"]["sem"], rel=1e-9)
    assert a["err_nu_plus"] == pytest.approx(c_e * 2.0, rel=1e-9)
    assert a["err_nu_e"] == pytest.approx(c_e * abs(1 - sol) * NU_E_ERR, rel=1e-9)
    assert b["err_nu_plus"] is None
    assert out["t_plus_COM"]["sem"] > 0 and out["t_plus_SOL"]["sem"] > 0
    meta = out["_meta"]["frames"]
    assert meta["nu_e_cm3_per_mol"] == NU_E
    assert meta["nu_plus_cm3_per_mol"] == {"A": -9.0, "B": 1.1}
    assert meta["c_e_mol_per_L"] == pytest.approx(c_e * 1e3)
    # without an error of ν_e the part is null, never silently zero
    bare = transport(traj, 300.0, "Li", "PF6", nu_e_cm3_per_mol=NU_E)
    assert bare["t_plus_VOL_A"]["err_nu_e"] is None


def test_every_t_plus_names_its_frame() -> None:
    out = transport(_trajectory(), 300.0, "Li", "PF6", nu_e_cm3_per_mol=NU_E)
    keys = {k for k in out if k.startswith("t_plus")}
    assert keys == {"t_plus_COM", "t_plus_SOL", "t_plus_VOL_A", "t_plus_VOL_B"}


def test_com_values_and_scaled_charges() -> None:
    """The force-field charge scale cancels in every frame."""
    traj = _trajectory()
    formal = transport(traj, 300.0, "Li", "PF6", nu_e_cm3_per_mol=NU_E)
    scaled = transport(traj, 300.0, "Li", "PF6", charges="scaled", nu_e_cm3_per_mol=NU_E)
    for k in ("t_plus_COM", "t_plus_SOL", "t_plus_VOL_A", "t_plus_VOL_B"):
        assert scaled[k]["mean"] == pytest.approx(formal[k]["mean"], rel=1e-9)


def test_frames_that_do_not_exist_are_refused_or_explained() -> None:
    with pytest.raises(ValueError, match="positive"):
        transport(_trajectory(), 300.0, "Li", "PF6", nu_e_cm3_per_mol=-1.0)
    with pytest.raises(ValueError, match="nu_e_cm3_per_mol"):
        transport(_trajectory(), 300.0, "Li", "PF6", nu_e_err_cm3_per_mol=1.0)
    # unequal ion numbers (no 1:1 salt): COM only, with the reason; VOL on request is an error
    uneven = _trajectory(n_anions=N_IONS - 2)
    out = transport(uneven, 300.0, "Li", "PF6")
    assert "t_plus_COM" in out and "t_plus_SOL" not in out
    assert "1:1" in out["_meta"]["frames"]["not_reported"]
    with pytest.raises(ValueError, match="1:1"):
        transport(uneven, 300.0, "Li", "PF6", nu_e_cm3_per_mol=NU_E)
    # a charged third species is not a solvent
    charged = _trajectory(solvent_charge=1)
    out = transport(charged, 300.0, "Li", "PF6")
    assert "t_plus_SOL" not in out and "EC" in out["_meta"]["frames"]["not_reported"]
    # VOL needs solvent: c_e ν_e < 1
    with pytest.raises(ValueError, match="volume"):
        transport(_trajectory(), 300.0, "Li", "PF6", nu_e_cm3_per_mol=5000.0)


def test_cli_reports_the_frames(tmp_path, capsys) -> None:
    import h5py

    traj = _trajectory()
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as fh:
        p = fh.create_group("particles/all/position")
        p["step"], p["time"], p["value"] = traj.steps, traj.times_ps, traj.positions_nm
        e = fh.create_group("particles/all/box/edges")
        e["step"], e["time"], e["value"] = traj.steps, traj.times_ps, traj.boxes_nm
        fh.create_group("parameters/ucmd")["system_json"] = json.dumps(traj.system.doc)
    args = ["transport", str(path), "--temperature-K", "300"]
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert "t_plus_SOL" in out and "t_plus_VOL_A" not in out and "_blocks" not in out
    assert cli.main([*args, "--nu-e-cm3-per-mol", "55.7", "--nu-e-err-cm3-per-mol", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["_meta"]["frames"]["nu_e_cm3_per_mol"] == 55.7
    assert out["t_plus_VOL_A"]["err_nu_e"] > 0
    with pytest.raises(SystemExit):
        cli.main([*args, "--nu-e-err-cm3-per-mol", "3"])
