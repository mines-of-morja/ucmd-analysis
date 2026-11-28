"""Self-diffusion and Onsager transport coefficients (PLAN.md A1.4, A1.5).

- Self-diffusion: D = lim MSD(t)/(6t), fitted on a window where β(t) = d ln MSD/d ln t ≈ 1.
- Onsager coefficients (Fong et al., Macromolecules 53, 9503 (2020); Lehnert 2025 Eq. 3), in the
  barycentric frame:
  L_ij = 1/(6 k_B T V) · d/dt ⟨Σ_a Δr_a^(i)(t) · Σ_b Δr_b^(j)(t)⟩.
- Conductivity κ = e² Σ_ij z_i z_j L_ij; COM-frame transference number
  t₊ = (z₊² L₊₊ + z₊z₋ L₊₋)/Σ_ij z_i z_j L_ij; Nernst–Einstein κ_NE = e²/(V k_B T) Σ_i N_i z_i² D_i.
- Decomposition (calculation plan D1): L_ii = N_i D_i/(k_B T V) + L_ii^d, the self part from the
  per-ion MSD and the distinct part from the cross terms a ≠ b; L₊₋ is distinct only. Then
  κ = κ_NE + e²(z₊² L₊₊^d + z₋² L₋₋^d + 2 z₊z₋ L₊₋) exactly, block by block, with the same fit
  window. A negative distinct sum (ionicity < 1) means correlated ion motion lowers κ; which
  term dominates tells ion pairing (L₊₋ > 0) from like-ion anticorrelation.
- Finite size (optional, cubic boxes only): with a viscosity η, the self-diffusion coefficients
  are also reported corrected, D∞ = D_PBC + k_B T ξ/(6π η L) (Yeh & Hummer 2004). η is an input
  whose source the caller records; the collective L_ij are not corrected.
- Reference frames (frames.py; ADR 0013): every t₊ names its frame in its key. `t_plus_COM` always;
  `t_plus_SOL` for a 1:1 salt in neutral solvent; `t_plus_VOL_A` and `t_plus_VOL_B` (the two
  ion-volume conventions) when the partial molar volume ν_e of the salt is given.
- Charges: z are the formal charges (±1) unless `charges="scaled"`, then the force field's
  (scaled by ζ). Which one a reference used must be checked (validation plan, pitfall).

Uncertainties: the trajectory is split into blocks; every coefficient is computed per block and
reported as mean ± standard error.
"""

from __future__ import annotations

import numpy as np

from . import constants as c
from .fit import FitWindow, slope
from .frames import (
    ION_VOLUME_CONVENTIONS,
    SolventFrame,
    frame_t_plus,
    sol_from_com,
    solvent_frame,
    vol_from_sol,
)
from .io import Trajectory
from .molecules import barycentric
from .msd import cross_msd, msd_series, self_msd
from .uncertainty import blocks, mean_sem


def beta(lags_ps: np.ndarray, msd: np.ndarray) -> np.ndarray:
    """β(t) = d ln MSD / d ln t (1 in the diffusive regime)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(np.log(msd[1:]), np.log(lags_ps[1:]))


def _frame_interval(traj: Trajectory) -> np.ndarray:
    dt = np.diff(traj.times_ps)
    if len(dt) == 0 or not np.allclose(dt, dt[0], rtol=1e-6):
        raise ValueError("frames must be equally spaced")
    return dt


def _check_viscosity(traj: Trajectory, viscosity_Pa_s: float | None) -> None:
    """The Yeh–Hummer correction needs a positive η and a cubic box."""
    if viscosity_Pa_s is None:
        return
    if not viscosity_Pa_s > 0:
        raise ValueError("the viscosity must be positive")
    h = traj.boxes_nm
    off = h - np.einsum("fii->fi", h)[:, :, None] * np.eye(3)
    edges = np.einsum("fii->fi", h)
    if np.abs(off).max() > 1e-9 or np.ptp(edges, axis=1).max() > 1e-6 * edges.max():
        raise ValueError("the Yeh–Hummer correction here is for cubic boxes only")


def diffusion(traj: Trajectory, temperature_K: float, *, species: list[str] | None = None,
              n_blocks: int = 5, window: FitWindow | None = None,
              viscosity_Pa_s: float | None = None) -> dict:  # fmt: skip
    """Self-diffusion coefficient D of each species (default: all), from the molecules' centres of
    mass in the barycentric frame, with block errors; no charges are needed, so it serves pure
    solvents. With a viscosity the Yeh–Hummer value D_YH = D + k_B T ξ/(6π η L) is added (cubic
    boxes; η is an input whose source the caller records). Keys as in `transport`."""
    window = window or FitWindow()
    dt = _frame_interval(traj)
    _check_viscosity(traj, viscosity_Pa_s)
    sys = traj.system
    names = species or list(sys.species)
    unknown = [n for n in names if n not in sys.species]
    if unknown:
        raise ValueError(f"no species {unknown[0]!r}; the trajectory has {list(sys.species)}")
    kt = c.BOLTZMANN_J_PER_K * temperature_K
    results: dict[str, list[float]] = {}
    com = {n: barycentric(traj, sys.species[n]) for n in names}
    for block in blocks(len(traj), n_blocks):
        lags = np.arange(block.stop - block.start) * dt[0]
        edge_m = float(traj.volumes_nm3()[block].mean()) ** (1 / 3) * c.NM_IN_M
        for n in names:
            d = slope(lags, self_msd(com[n][block]), window) / 6.0 * c.NM2_PER_PS_IN_M2_PER_S
            results.setdefault(f"D_{n}_m2_per_s", []).append(d)
            if viscosity_Pa_s is not None:
                shift = kt * c.YEH_HUMMER_XI_CUBIC / (6 * np.pi * viscosity_Pa_s * edge_m)
                results.setdefault(f"D_{n}_YH_m2_per_s", []).append(d + shift)
    out = {k: dict(zip(("mean", "sem"), mean_sem(v), strict=True)) for k, v in results.items()}
    out["_meta"] = {
        "temperature_K": temperature_K,
        "frames": len(traj),
        "frame_interval_ps": float(dt[0]),
        "blocks": n_blocks,
        "fit_window_fraction": [window.start, window.stop],
        "viscosity_Pa_s": viscosity_Pa_s,
    }
    return out


def transport(traj: Trajectory, temperature_K: float, cation: str, anion: str, *,
              n_blocks: int = 5, window: FitWindow | None = None,
              charges: str = "formal", viscosity_Pa_s: float | None = None,
              nu_e_cm3_per_mol: float | None = None, nu_e_err_cm3_per_mol: float | None = None,
              keep_blocks: bool = False) -> dict:  # fmt: skip
    """D of every species, L_ij, κ, t₊ (COM, SOL and, with ν_e, VOL frame), κ_NE and the ionicity,
    with block errors.

    ν_e is the partial molar volume of the salt (cm³/mol), its error optional; the caller records
    the source. A t₊ in the VOL frame carries its error parts separately: `sem` (blocks),
    `err_nu_plus` (stated for the convention, else null) and `err_nu_e` (null without an error of
    ν_e). The parts are partial derivatives times the input error; they are not added up.
    `keep_blocks` adds the per-block values under `_blocks`."""
    window = window or FitWindow()
    dt = _frame_interval(traj)
    _check_viscosity(traj, viscosity_Pa_s)
    if nu_e_cm3_per_mol is not None and not nu_e_cm3_per_mol > 0:
        raise ValueError("nu_e must be positive")
    if nu_e_err_cm3_per_mol is not None:
        if nu_e_cm3_per_mol is None:
            raise ValueError("an error of ν_e needs nu_e_cm3_per_mol")
        if not nu_e_err_cm3_per_mol > 0:
            raise ValueError("the error of nu_e must be positive")
    sys = traj.system
    names = list(sys.species)
    ions = (cation, anion)
    z = {
        n: (sys.species[n].formal_charge if charges == "formal" else sys.species[n].charge)
        for n in ions
    }  # fmt: skip
    for n in ions:
        if sys.species[n].count == 0 or abs(z[n]) < 1e-9:
            raise ValueError(
                f"{n} is neutral (charge {z[n]:+.3f} e): conductivity and transference numbers "
                "need a charged cation and anion; for a neutral species use `diffusion` "
                "(ucmd-analyze diffusion)"
            )
    solvent = solvent_frame(sys, cation, anion)
    if nu_e_cm3_per_mol is not None and not isinstance(solvent, SolventFrame):
        raise ValueError(f"no VOL frame: {solvent}")
    com = {n: barycentric(traj, sys.species[n]) for n in names}
    kt = c.BOLTZMANN_J_PER_K * temperature_K
    results: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        results.setdefault(key, []).append(value)

    for block in blocks(len(traj), n_blocks):
        lags = (np.arange(block.stop - block.start)) * dt[0]
        volume_m3 = float(traj.volumes_nm3()[block].mean()) * c.NM_IN_M**3
        d = {}
        for n in names:
            m = self_msd(com[n][block])  # nm²
            d[n] = slope(lags, m, window) / 6.0 * c.NM2_PER_PS_IN_M2_PER_S
            add(f"D_{n}_m2_per_s", d[n])
            if viscosity_Pa_s is not None:
                edge_m = volume_m3 ** (1 / 3)
                shift = kt * c.YEH_HUMMER_XI_CUBIC / (6 * np.pi * viscosity_Pa_s * edge_m)
                add(f"D_{n}_YH_m2_per_s", d[n] + shift)
        sums = {n: com[n][block].sum(axis=1) for n in ions}  # (frames, 3) nm
        lij, lself = {}, {}
        for a, b in ((cation, cation), (anion, anion), (cation, anion)):
            m = msd_series(sums[a]) if a == b else cross_msd(sums[a], sums[b])
            s = slope(lags, m, window) * c.NM2_PER_PS_IN_M2_PER_S  # m²/s
            lij[(a, b)] = s / (6.0 * kt * volume_m3)  # 1/(J m s)
            if a == b:  # self part: Σ_a |Δr_a|², the same series D_a comes from
                s = slope(lags, msd_series(com[a][block]), window) * c.NM2_PER_PS_IN_M2_PER_S
                lself[a] = s / (6.0 * kt * volume_m3)
        lpp, lmm, lpm = lij[(cation, cation)], lij[(anion, anion)], lij[(cation, anion)]
        zp, zm = z[cation], z[anion]
        e2 = c.ELEMENTARY_CHARGE_C**2
        total = zp * zp * lpp + zm * zm * lmm + 2 * zp * zm * lpm
        kappa = e2 * total  # S/m
        kappa_ne = (
            e2 / (volume_m3 * kt) * sum(sys.species[n].count * z[n] ** 2 * d[n] for n in ions)
        )
        add("L_pp", lpp)
        add("L_mm", lmm)
        add("L_pm", lpm)
        add("kappa_S_per_m", kappa)
        add("kappa_NE_S_per_m", kappa_ne)
        add("ionicity", kappa / kappa_ne)
        add("t_plus_COM", (zp * zp * lpp + zp * zm * lpm) / total)
        if isinstance(solvent, SolventFrame):
            r_sol = solvent.displacement(com)[block]
            sol = frame_t_plus(sums[cation] - sys.species[cation].count * r_sol,
                               sums[anion] - sys.species[anion].count * r_sol,
                               lags, window).t_plus  # fmt: skip
            add("t_plus_SOL", sol)
            eq2 = sol_from_com(results["t_plus_COM"][-1], solvent.omega_minus, solvent.omega_0)
            add("_sol_minus_eq2", sol - eq2)
            c_e = (
                sys.species[cation].count
                / c.AVOGADRO_PER_MOL
                / (volume_m3 / c.NM_IN_M**3 * c.NM3_IN_CM3)
            )
            add("_c_e_mol_per_cm3", c_e)
            if nu_e_cm3_per_mol is not None:
                if not c_e * nu_e_cm3_per_mol < 1:
                    raise ValueError(
                        f"nu_e = {nu_e_cm3_per_mol} cm³/mol gives the salt the whole volume "
                        f"(c_e nu_e = {c_e * nu_e_cm3_per_mol:.3f})"
                    )
                for name, conv in ION_VOLUME_CONVENTIONS.items():
                    add(f"t_plus_VOL_{name}",
                        vol_from_sol(sol, c_e, nu_e_cm3_per_mol,
                                     conv.nu_plus_cm3_per_mol))  # fmt: skip
        # D1: κ = κ_self,+ + κ_self,− + κ_d,++ + κ_d,−− + κ_d,+− (S/m)
        add("kappa_self_cation_S_per_m", e2 * zp * zp * lself[cation])
        add("kappa_self_anion_S_per_m", e2 * zm * zm * lself[anion])
        add("kappa_distinct_pp_S_per_m", e2 * zp * zp * (lpp - lself[cation]))
        add("kappa_distinct_mm_S_per_m", e2 * zm * zm * (lmm - lself[anion]))
        add("kappa_distinct_pm_S_per_m", e2 * 2 * zp * zm * lpm)
    out = {k: dict(zip(("mean", "sem"), mean_sem(v), strict=True)) for k, v in results.items()
           if not k.startswith("_")}  # fmt: skip
    out["_meta"] = {
        "temperature_K": temperature_K,
        "frames": len(traj),
        "frame_interval_ps": float(dt[0]),
        "blocks": n_blocks,
        "fit_window_fraction": [window.start, window.stop],
        "charges": charges,
        "viscosity_Pa_s": viscosity_Pa_s,
        "z": z,
    }
    out["_meta"]["frames"] = _frames_meta(
        out, results, solvent, nu_e_cm3_per_mol, nu_e_err_cm3_per_mol
    )
    if keep_blocks:
        out["_blocks"] = {k: v for k, v in results.items() if not k.startswith("_")}
    return out


def _frames_meta(out: dict, results: dict, solvent: SolventFrame | str,
                 nu_e: float | None, nu_e_err: float | None) -> dict:  # fmt: skip
    """The inputs of the frames, and the error parts of every t₊^VOL (added to `out`)."""
    if not isinstance(solvent, SolventFrame):
        return {"not_reported": f"t_plus_SOL and t_plus_VOL: {solvent}"}
    c_e = float(np.mean(results["_c_e_mol_per_cm3"]))
    meta = {
        "solvent": list(solvent.names),
        "omega_minus": solvent.omega_minus,
        "omega_0": solvent.omega_0,
        "max_abs_SOL_minus_Eq2": float(np.max(np.abs(results["_sol_minus_eq2"]))),
        "c_e_mol_per_L": c_e * 1e3,
    }
    if nu_e is None:
        meta["not_reported"] = "t_plus_VOL: needs nu_e_cm3_per_mol"
        return meta
    for name, conv in ION_VOLUME_CONVENTIONS.items():
        value = out[f"t_plus_VOL_{name}"]
        err = conv.err_cm3_per_mol
        value["err_nu_plus"] = None if err is None else c_e * err
        dt_dnu_e = c_e * abs(1 - out["t_plus_SOL"]["mean"])
        value["err_nu_e"] = None if nu_e_err is None else dt_dnu_e * nu_e_err
    meta["nu_e_cm3_per_mol"] = nu_e
    meta["nu_e_err_cm3_per_mol"] = nu_e_err
    meta["nu_plus_cm3_per_mol"] = {
        k: v.nu_plus_cm3_per_mol for k, v in ION_VOLUME_CONVENTIONS.items()
    }
    meta["nu_plus_source"] = {k: v.source for k, v in ION_VOLUME_CONVENTIONS.items()}
    meta["central_convention"] = "A"
    return meta
