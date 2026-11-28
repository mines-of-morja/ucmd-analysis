"""Shear viscosity by Green–Kubo from the pressure tensor (calculation plan A6, spec 0.1.1).

η = V/(k_B T) ∫₀^∞ ⟨P_ab(0) P_ab(t)⟩ dt, averaged over the five independent components of the
traceless symmetric tensor: P_xy, P_xz, P_yz, (P_xx − P_yy)/2 and (P_yy − P_zz)/2. In an
isotropic fluid each has the same correlation integral.

- **Input:** the thermo table written with `[output.thermo] pressure_tensor = true` (bar).
- **Sampling:** the table must resolve the fastest oscillation of the tensor. With flexible bonds
  that is about 10 fs, so every 1–2 steps of 1 fs.
- **Integration:** the integral is evaluated up to `max_lag_ps`. The running integral η(t)
  should have a plateau there; it is returned (subsampled) so that this can be checked.
- **Errors:** η as mean ± sem over blocks, each with its own correlation function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import constants as c
from .uncertainty import blocks, mean_sem

COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def read_tensor(path: Path | str, skip_ps: float = 0.0) -> tuple[np.ndarray, dict, float]:
    """(time_ps, {component: bar}, mean volume nm³) from a thermo table."""
    with Path(path).open() as fh:
        names = fh.readline().lstrip("# ").strip().split("\t")
    missing = [k for k in COMPONENTS if f"pressure_{k}_bar" not in names]
    if missing:
        raise ValueError(
            f"{path}: no pressure-tensor columns ({', '.join(missing)}); run with "
            "[output.thermo] pressure_tensor = true (spec 0.1.1)"
        )
    rows = np.loadtxt(path, ndmin=2)
    t = rows[:, names.index("time_ps")]
    w = t >= skip_ps
    p = {k: rows[w, names.index(f"pressure_{k}_bar")] for k in COMPONENTS}
    return t[w], p, float(np.mean(rows[w, names.index("volume_nm3")]))


def shear_components(p: dict) -> np.ndarray:
    """(5, n) independent shear components of the traceless symmetric tensor, in bar."""
    return np.stack([p["xy"], p["xz"], p["yz"], 0.5 * (p["xx"] - p["yy"]),
                     0.5 * (p["yy"] - p["zz"])])  # fmt: skip


def _acf(x: np.ndarray, n_lag: int) -> np.ndarray:
    """Mean autocorrelation of the rows of x (FFT, unbiased), lags 0..n_lag−1."""
    n = x.shape[1]
    f = np.fft.rfft(x - x.mean(axis=1, keepdims=True), n=2 * n, axis=1)
    acf = np.fft.irfft(f * f.conj(), n=2 * n, axis=1)[:, :n_lag].real
    return acf.mean(axis=0) / (n - np.arange(n_lag))


def _eta(shear: np.ndarray, dt_ps: float, volume_nm3: float, temperature_K: float,
         n_lag: int) -> np.ndarray:  # fmt: skip
    """Running Green–Kubo integral η(t) in Pa·s."""
    acf = _acf(shear, n_lag) * c.BAR_IN_PA**2  # Pa²
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (acf[1:] + acf[:-1]))]) * dt_ps * c.PS_IN_S
    return volume_nm3 * c.NM_IN_M**3 / (c.BOLTZMANN_J_PER_K * temperature_K) * integral


def green_kubo(shear: np.ndarray, dt_ps: float, volume_nm3: float, temperature_K: float, *,
               n_blocks: int = 5, max_lag_ps: float = 5.0) -> dict:  # fmt: skip
    n_lag = round(max_lag_ps / dt_ps) + 1
    if shear.shape[1] // n_blocks < 4 * n_lag:
        raise ValueError(
            f"blocks of {shear.shape[1] // n_blocks} samples are too short for lags up to "
            f"{max_lag_ps} ps ({n_lag} samples); use fewer blocks or a shorter max lag"
        )
    running = _eta(shear, dt_ps, volume_nm3, temperature_K, n_lag)
    per = [_eta(shear[:, s], dt_ps, volume_nm3, temperature_K, n_lag)[-1]
           for s in blocks(shear.shape[1], n_blocks)]  # fmt: skip
    mean, sem = mean_sem(per)
    step = max(1, n_lag // 200)
    return {
        "viscosity_Pa_s": {"mean": mean, "sem": sem, "blocks": n_blocks, "values": per},
        "viscosity_whole_series_Pa_s": float(running[-1]),
        "running_lag_ps": (np.arange(n_lag)[::step] * dt_ps).round(6).tolist(),
        "running_viscosity_Pa_s": running[::step].tolist(),
        "max_lag_ps": max_lag_ps,
        "components": int(shear.shape[0]),
        "samples": int(shear.shape[1]),
        "dt_ps": dt_ps,
        "volume_nm3": volume_nm3,
        "temperature_K": temperature_K,
    }
