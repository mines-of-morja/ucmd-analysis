"""Green–Kubo form of the Onsager coefficients, a cross-check of `transport` (A1.5; design
A1.5-green-kubo.md of UnitcellMD, option A).

L_ij = 1/(3 k_B T V) ∫₀^∞ ⟨J_i(0)·J_j(t)⟩ dt with J_i = Σ_a v_a, the summed centre-of-mass velocity
of the molecules of species i in the barycentric frame. It has the normalisation of the Einstein
form (Fong et al. 2020; Lehnert 2025 Eq. 3), because d/dt ⟨ΔR_i·ΔR_j⟩ = 2∫₀ᵗ⟨J_i(0)·J_j(t')⟩dt'.

- **Not an independent estimate.** Both forms of the same trajectory agree in the limit of
  continuous sampling: the check tests the implementation, the sampling interval and the choice
  of the regime, not the statistics.
- **Input:** H5MD velocities (`[output.trajectory] velocities = true`) at a stride that resolves the
  velocity correlation. The stride study re-evaluates the same run at 2× and 4× stride.
- **Window:** the value is the mean of the running integral over `plateau_ps`, which has no default:
  it is fixed before the comparison. `plateau` reports whether the integral is flat there (the
  second half of the window against the first, within 2 SEM). A noisy term can pass that test
  without having a plateau; the running integral is in the output.
- **Einstein values, two windows:** `einstein` uses the fit window of `transport`;
  `einstein_same_window` fits the collective MSD over `plateau_ps`. Only the second is a sharp
  test, because long lags of a collective MSD are mostly noise; `difference` is taken to it.
- **Errors:** every quantity per block, as mean ± SEM. The cross term uses both time orders.
"""

from __future__ import annotations

import numpy as np

from . import constants as c
from .fit import FitWindow
from .io import Molecules, Trajectory
from .transport import transport
from .uncertainty import blocks, mean_sem

_KEYS = ("L_pp", "L_mm", "L_pm", "kappa_S_per_m", "t_plus_COM")


def species_current(traj: Trajectory, mol: Molecules) -> np.ndarray:
    """J_i(t), (frames, 3) in nm/ps: summed molecular velocities relative to the barycentre."""
    if traj.velocities_nm_per_ps is None:
        raise ValueError("the trajectory carries no velocities (read_h5md(..., velocities=True))")
    v = traj.velocities_nm_per_ps
    m = traj.system.masses
    bary = np.einsum("fad,a->fd", v, m / m.sum())
    w = mol.masses / mol.masses.sum()
    total = np.zeros((len(v), 3))
    for k, weight in enumerate(w):  # per template atom: no copy of all velocities at once
        total += weight * v[:, mol.atoms[:, k], :].sum(axis=1)
    return total - mol.count * bary


def running_integral(a: np.ndarray, b: np.ndarray, dt_ps: float, n_lag: int) -> np.ndarray:
    """∫₀^t ⟨a(0)·b(t')⟩ dt' (trapezoid) at lags 0 … n_lag−1, with both time orders averaged.

    The mean is not subtracted: an equilibrium current in the barycentric frame has mean zero."""
    n = len(a)
    fa, fb = np.fft.rfft(a, n=2 * n, axis=0), np.fft.rfft(b, n=2 * n, axis=0)
    # the real part of the spectrum is the part even in time: the average of both time orders
    corr = np.fft.irfft((fa.conj() * fb).real.sum(axis=1), n=2 * n)[:n_lag]
    corr = corr / (n - np.arange(n_lag))
    return np.concatenate([[0.0], np.cumsum(0.5 * (corr[1:] + corr[:-1]))]) * dt_ps


def _derived(lpp: float, lmm: float, lpm: float, zp: float, zm: float) -> dict[str, float]:
    total = zp * zp * lpp + zm * zm * lmm + 2 * zp * zm * lpm
    return {
        "L_pp": lpp,
        "L_mm": lmm,
        "L_pm": lpm,
        "kappa_S_per_m": c.ELEMENTARY_CHARGE_C**2 * total,
        "t_plus_COM": (zp * zp * lpp + zp * zm * lpm) / total,
    }


def _stat(values) -> dict[str, float]:
    return dict(zip(("mean", "sem"), mean_sem(values), strict=True))


def onsager_green_kubo(traj: Trajectory, temperature_K: float, cation: str, anion: str, *,
                       plateau_ps: tuple[float, float], max_lag_ps: float, n_blocks: int = 5,
                       window: FitWindow | None = None,
                       charges: str = "formal") -> dict:  # fmt: skip
    """L₊₊, L₋₋, L₊₋, κ, t₊ (COM) from velocities, next to the Einstein form of `transport` on the
    same blocks. `window` is the Einstein fit window."""
    window = window or FitWindow()
    if traj.velocities_nm_per_ps is None:
        raise ValueError("the trajectory carries no velocities (read_h5md(..., velocities=True))")
    ein = transport(traj, temperature_K, cation, anion, n_blocks=n_blocks, window=window,
                    charges=charges, keep_blocks=True)  # fmt: skip
    z = ein["_meta"]["z"]
    zp, zm = z[cation], z[anion]
    dt = ein["_meta"]["frame_interval_ps"]
    n_lag = round(max_lag_ps / dt) + 1
    lo, hi = plateau_ps
    if not 0 <= lo < hi <= max_lag_ps * (1 + 1e-9):
        raise ValueError(f"plateau window {lo}–{hi} ps must lie within 0–{max_lag_ps} ps")
    size = len(traj) // n_blocks
    if size < 4 * n_lag:
        raise ValueError(
            f"blocks of {size} frames are too short for lags up to {max_lag_ps} ps ({n_lag} "
            "frames): at least four times as many are needed"
        )
    sp = traj.system.species
    j = {n: species_current(traj, sp[n]) for n in (cation, anion)}
    pairs = {"L_pp": (cation, cation), "L_mm": (anion, anion), "L_pm": (cation, anion)}
    volumes = traj.volumes_nm3()

    def integrals(stride: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Lags (ps) and, per coefficient, the running L(t) of every block (1/(J m s))."""
        k = (n_lag - 1) // stride + 1
        out: dict[str, list] = {name: [] for name in pairs}
        for block in blocks(len(traj), n_blocks):
            v_m3 = float(volumes[block].mean()) * c.NM_IN_M**3
            pre = c.NM2_PER_PS_IN_M2_PER_S / (3 * c.BOLTZMANN_J_PER_K * temperature_K * v_m3)
            for name, (a, b) in pairs.items():
                ja, jb = j[a][block][::stride], j[b][block][::stride]
                out[name].append(pre * running_integral(ja, jb, dt * stride, k))
        return np.arange(k) * dt * stride, {n: np.array(v) for n, v in out.items()}

    def window_mean(lags: np.ndarray, run: np.ndarray, a: float, b: float) -> np.ndarray:
        inside = (lags >= a - 1e-9) & (lags <= b + 1e-9)
        if not inside.any():
            raise ValueError(f"plateau window {a}–{b} ps holds no sample at this stride")
        return run[:, inside].mean(axis=1)

    lags, run = integrals(1)
    per = {n: window_mean(lags, r, lo, hi) for n, r in run.items()}
    rows = [_derived(*(per[n][k] for n in pairs), zp, zm) for k in range(n_blocks)]
    gk_blocks = {key: [row[key] for row in rows] for key in _KEYS}
    ein_blocks = {key: ein["_blocks"][key] for key in _KEYS}
    span = size * dt
    same = transport(traj, temperature_K, cation, anion, n_blocks=n_blocks, charges=charges,
                     window=FitWindow(lo / span, hi / span), keep_blocks=True)  # fmt: skip
    same_blocks = {key: same["_blocks"][key] for key in _KEYS}
    mid = 0.5 * (lo + hi)
    plateau = {}
    for name, r in run.items():
        drift = _stat(window_mean(lags, r, mid, hi) - window_mean(lags, r, lo, mid))
        plateau[name] = {"drift": drift, "flat": bool(abs(drift["mean"]) <= 2 * drift["sem"])}
    study: dict[str, dict[str, float]] = {name: {} for name in pairs}
    for stride in (1, 2, 4):
        s_lags, s_run = (lags, run) if stride == 1 else integrals(stride)
        for name, r in s_run.items():
            study[name][str(stride)] = float(window_mean(s_lags, r, lo, hi).mean())
    step = max(1, n_lag // 200)
    return {
        "green_kubo": {k: _stat(v) for k, v in gk_blocks.items()},
        "einstein": {k: _stat(v) for k, v in ein_blocks.items()},
        "einstein_same_window": {k: _stat(v) for k, v in same_blocks.items()},
        "difference": {
            k: _stat(np.subtract(gk_blocks[k], same_blocks[k])) for k in _KEYS
        },  # Green–Kubo − Einstein over the plateau window, paired by block
        "plateau": plateau,
        "stride_study": study,
        "running": {
            name: {
                "lag_ps": lags[::step].tolist(),
                "mean": r.mean(axis=0)[::step].tolist(),
                "sem": (r.std(axis=0, ddof=1) / np.sqrt(n_blocks))[::step].tolist()
                if n_blocks > 1
                else [None] * len(lags[::step]),
            }
            for name, r in run.items()
        },
        "_meta": {
            "temperature_K": temperature_K,
            "frames": len(traj),
            "frame_interval_ps": dt,
            "blocks": n_blocks,
            "plateau_ps": [lo, hi],
            "max_lag_ps": max_lag_ps,
            "einstein_fit_window_fraction": [window.start, window.stop],
            "charges": charges,
            "z": z,
            "note": "both forms use the same trajectory: agreement is a test of the analysis, "
            "not an independent estimate",
        },
    }
