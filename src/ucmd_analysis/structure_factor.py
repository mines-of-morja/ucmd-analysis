"""Partial structure factors at small k and their k → 0 limit (A1.7, design/A1.7-tdf.md).

S_αβ(k) = ⟨ρ_α(k) ρ_β(−k)⟩ / √(N_α N_β), ρ_α(k) = Σ_a exp(i k·r_a), from molecular centres of mass,
on the reciprocal lattice of the mean box and in scaled coordinates (Cheng 2022, Eqs. 12, 17), so
that NPT runs can be used. k → 0 itself is never sampled: N is fixed in the box.

Limits (kept as k² fits on the inverse matrix, S⁻¹ = A + k² L, weighted by the number of modes):

- `scalar`: each element alone, 1/S_αβ linear in k² (Cheng 2022, Eq. 18);
- `matrix`: the Ornstein–Zernike matrix (Wang et al. 2026, Part II, Eq. 17);
- `coulomb`: the bare-Coulomb matrix (their Eqs. 20–22) for charged species kept apart,
  S⁻¹ = A + k² L + w wᵀ κ²/(wᵀw k²), w = √x z, limit A⁻¹ − A⁻¹wwᵀA⁻¹/(wᵀA⁻¹w). κ² is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkSeries:
    """ρ_α(k, t) for every species, frame and half-space reciprocal vector."""

    rho: np.ndarray  # complex (frames, species, vectors)
    k2_nm2: np.ndarray  # |k|² in nm⁻², (vectors,), ascending
    counts: np.ndarray  # particles per species


def reciprocal_vectors(box_nm: np.ndarray, k_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer triples (half space, without 0) with |k| ≤ k_max (nm⁻¹), and |k|², ascending."""
    g = 2 * np.pi * np.linalg.inv(box_nm).T  # rows: reciprocal vectors of a, b, c
    n_max = np.floor(k_max * np.linalg.norm(box_nm, axis=1) / (2 * np.pi)).astype(int)
    grid = np.stack(
        np.meshgrid(*[np.arange(-m, m + 1) for m in n_max], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    half = [tuple(t) > (0, 0, 0) for t in grid.tolist()]
    grid = grid[half]
    k2 = np.sum((grid @ g) ** 2, axis=1)
    keep = k2 <= k_max**2 * (1 + 1e-12)
    grid, k2 = grid[keep], k2[keep]
    order = np.argsort(k2, kind="stable")
    return grid[order], k2[order]


def density_modes(positions: list[np.ndarray], boxes_nm: np.ndarray, k_max: float) -> SkSeries:
    """ρ_α(k, t) of each species; positions are (frames, particles, 3) in nm (any image)."""
    mean_box = boxes_nm.mean(axis=0)
    triples, k2 = reciprocal_vectors(mean_box, k_max)
    if len(triples) == 0:
        smallest = 2 * np.pi / np.linalg.norm(mean_box, axis=1).max()
        raise ValueError(f"no reciprocal vector below {k_max} nm⁻¹; the smallest is {smallest:.3f}")
    n_max = np.abs(triples).max(axis=0)
    frames = len(boxes_nm)
    rho = np.empty((frames, len(positions), len(triples)), dtype=complex)
    for f in range(frames):
        inv = np.linalg.inv(boxes_nm[f])
        for s, x in enumerate(positions):
            frac = x[f] @ inv
            axis = []
            for d in range(3):
                p = np.exp(2j * np.pi * np.outer(np.arange(-n_max[d], n_max[d] + 1), frac[:, d]))
                axis.append(p[triples[:, d] + n_max[d]])  # (vectors, particles)
            rho[f, s] = (axis[0] * axis[1] * axis[2]).sum(axis=1)
    return SkSeries(rho, k2, np.array([x.shape[1] for x in positions], dtype=float))


def partial_sk(series: SkSeries, frames: slice = slice(None)) -> tuple[np.ndarray, ...]:
    """(k², modes, S) per shell of equal |k|: S has shape (shells, species, species), `modes` is
    the number of half-space vectors (each a complex, so two real modes)."""
    r = series.rho[frames]
    s = np.einsum("fak,fbk->kab", r, r.conj()).real / len(r)
    s /= np.sqrt(np.outer(series.counts, series.counts))
    shell = np.concatenate([[0], np.cumsum(np.diff(series.k2_nm2) > 1e-6 * series.k2_nm2[:-1])])
    n = shell[-1] + 1
    modes = np.bincount(shell, minlength=n).astype(float)
    k2 = np.bincount(shell, weights=series.k2_nm2, minlength=n) / modes
    out = np.zeros((n, *s.shape[1:]))
    np.add.at(out, shell, s)
    return k2, modes, out / modes[:, None, None]


def _linear(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares of y (samples, …) on the columns of x (samples, terms)."""
    sw = np.sqrt(w)
    flat = (y.reshape(len(y), -1)) * sw[:, None]
    coef = np.linalg.lstsq(x * sw[:, None], flat, rcond=None)[0]
    return coef.reshape(x.shape[1], *y.shape[1:])


def s0(k2: np.ndarray, modes: np.ndarray, sk: np.ndarray, *, form: str = "matrix",
       charges: np.ndarray | None = None, x: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """The k → 0 limit of S from all shells given; `charges` (e) and the number fractions `x`
    belong to the `coulomb` form."""
    design = np.stack([np.ones_like(k2), k2], axis=1)
    if len(k2) < 3:
        raise ValueError(f"{len(k2)} k-shells are too few for a fit; raise k_max or the box")
    if form == "scalar":
        return 1.0 / _linear(design, 1.0 / sk, modes)[0]
    inv = np.linalg.inv(sk)
    if form == "matrix":
        return np.linalg.inv(_linear(design, inv, modes)[0])
    if form != "coulomb":
        raise ValueError(f"unknown form {form!r}: scalar, matrix or coulomb")
    if charges is None or x is None or len(charges) != sk.shape[1]:
        raise ValueError("the coulomb form needs the charge and number fraction of every species")
    w = np.sqrt(x) * np.asarray(charges, dtype=float)
    unit = np.outer(w, w) / (w @ w)
    # the 1/k² term lives on one fixed matrix: fit it as a third column, keep its projection
    three = np.stack([np.ones_like(k2), k2, 1.0 / k2], axis=1)
    coef = _linear(three, inv, modes)
    kappa2 = float(np.sum(coef[2] * unit))
    if kappa2 <= 0:
        raise ValueError("no screening found at small k: the fitted κ² is not positive")
    rest = inv - kappa2 * unit[None] / k2[:, None, None]
    a_inv = np.linalg.inv(_linear(design, rest, modes)[0])
    return a_inv - a_inv @ np.outer(w, w) @ a_inv / (w @ a_inv @ w)
