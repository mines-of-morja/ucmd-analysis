"""Analysis of a17-lj (README.md; criterion fixed there): S0_AB of Cheng 2022.

python analyze.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from make_system import SCALE, SIGMA_NM

from ucmd_analysis.io import read_h5md
from ucmd_analysis.structure_factor import SkSeries, density_modes, partial_sk, s0
from ucmd_analysis.thermodynamic_factor import relaxation_time_ps
from ucmd_analysis.uncertainty import blocks

TARGET, TARGET_ERR = -0.439, 0.004  # Cheng 2022, Fig. 2a, N = 4000
N_BLOCKS = 10
FIT_2PI_PER_SIGMA = 0.2  # Cheng 2022, p. 3
SCAN = (0.15, 0.2, 0.25, 0.3)
MIN_BLOCK_PER_TAU = 5.0
TEMPERATURE_K = 144.0  # T = 1.2, for the record only


def limit(series: SkSeries, frames, kc: float, form: str) -> np.ndarray:
    k2, modes, sk = partial_sk(series, frames)
    keep = k2 <= (2 * np.pi * kc / SIGMA_NM) ** 2
    return s0(k2[keep], modes[keep], sk[keep], form=form)


def with_jackknife(series: SkSeries, kc: float, form: str, f) -> tuple[float, float]:
    """f(S0 of all frames) and its standard error, leaving out one block at a time."""
    n = len(series.rho)
    used = (n // N_BLOCKS) * N_BLOCKS
    all_frames = np.arange(used)
    value = f(limit(series, all_frames, kc, form))
    part = []
    for b in blocks(n, N_BLOCKS):
        rest = np.setdiff1d(all_frames, all_frames[b])
        part.append(f(limit(series, rest, kc, form)))
    part = np.array(part)
    err = np.sqrt((N_BLOCKS - 1) / N_BLOCKS * np.sum((part - part.mean()) ** 2))
    return float(value), float(err)


def s_ab(s: np.ndarray) -> float:
    return s[0, 1]


def dmu(s: np.ndarray) -> float:
    """d(beta mu_A)/d ln x_A at x_A = 0.5 (Cheng Eq. 21): 1 for an ideal mixture."""
    return 1 / (0.5 * s[0, 0] + 0.5 * s[1, 1] - s[0, 1])


def compressibility(s: np.ndarray) -> float:
    """rho kB T kappa_T = 1/(1^T S^-1 1) scaled to equal numbers: c^T B^-1 c = (rho/2) sum S^-1."""
    return 2 / np.sum(np.linalg.inv(s))


def analyze(out: Path, name: str) -> dict:
    traj = read_h5md(out / name / "trajectory.h5")
    series = density_modes(
        [traj.positions_nm[:, traj.system.species[s].atoms.ravel()] for s in ("A", "B")],
        traj.boxes_nm,
        2 * np.pi * max(SCAN) / SIGMA_NM,
    )
    dt = float(np.mean(np.diff(traj.times_ps)))
    tau = max(relaxation_time_ps(series, dt, s) for s in (0, 1))
    block = len(traj) // N_BLOCKS * dt
    v = traj.volumes_nm3()
    n = sum(m.count for m in traj.system.species.values())
    rho = n * SIGMA_NM**3 / v.mean()
    res = {
        "frames": len(traj),
        "provenance": {k: str(x) for k, x in traj.provenance.items()},
        "relaxation_time_ps": tau,
        "block_ps": block,
        "block_long_enough": bool(block >= MIN_BLOCK_PER_TAU * tau),
        "rho_reduced": float(rho),
        "S0_AB": {},
        "dmu_dlnx": {},
    }
    for kc in SCAN:
        for form in ("scalar", "matrix"):
            res["S0_AB"][f"{form}_{kc}"] = with_jackknife(series, kc, form, s_ab)
            res["dmu_dlnx"][f"{form}_{kc}"] = with_jackknife(series, kc, form, dmu)
    res["rho_kT_kappa_from_S0"] = with_jackknife(
        series, FIT_2PI_PER_SIGMA, "matrix", compressibility
    )
    if v.std() > 1e-9 * v.mean():
        pieces = [v[b] for b in blocks(len(v), N_BLOCKS)]
        value = v.var() / v.mean() * n / v.mean()
        est = np.array([(p.var() / p.mean() * n / p.mean()) for p in pieces])
        res["rho_kT_kappa_from_volume"] = [float(value), float(est.std(ddof=1) / np.sqrt(N_BLOCKS))]
    return res


def verdict(npt: dict) -> dict:
    value, err = npt["S0_AB"][f"scalar_{FIT_2PI_PER_SIGMA}"]
    bound = 2 * float(np.hypot(err, TARGET_ERR))
    if not npt["block_long_enough"]:
        return {"verdict": "not judged: continue the run", "value": value, "se": err}
    ok = abs(value - TARGET) <= bound
    return {"verdict": "pass" if ok else "FAIL", "value": value, "se": err, "bound": bound,
            "difference": value - TARGET}  # fmt: skip


def main() -> None:
    out = Path(sys.argv[1])
    result: dict = {"scale": SCALE, "temperature_K": TEMPERATURE_K}
    result["npt"] = analyze(out, "npt")
    result["criterion"] = verdict(result["npt"])
    if (out / "nvt" / "trajectory.h5").exists():
        result["nvt"] = analyze(out, "nvt")
        key = f"scalar_{FIT_2PI_PER_SIGMA}"
        a, b = result["npt"]["S0_AB"][key], result["nvt"]["S0_AB"][key]
        result["nvt_minus_npt_in_se"] = (b[0] - a[0]) / float(np.hypot(a[1], b[1]))
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
