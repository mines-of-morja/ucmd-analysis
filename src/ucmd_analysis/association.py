"""Ion association, solvation shells and residence times (calculation plan D4, D5).

Each molecule is represented by one atom: Li for Li⁺, P for PF₆⁻, the carbonyl O for a
carbonate. Two molecules are in contact when these atoms are closer than r_cut (minimum image,
triclinic cells as in `structure`). r_cut is an analysis choice, normally the first minimum of
the corresponding g(r) (`structure.coordination`), and is always reported with the result.

- **Clusters (D4):** cations and anions are the nodes of a graph whose edges are cation–anion
  contacts. Each connected component is a cluster (n₊, n₋) with charge n₊z₊ + n₋z₋: free ions,
  contact pairs (1, 1), triple ions (2, 1) and (1, 2), larger aggregates.
- **Shell composition:** for each cation, the number of contacts with each partner group in the
  same frame; populations of the combinations, e.g. "EC:O:1=2 EMC:O:1=1 PF6:P=1".
- **Residence times (D5):** h_ij(t) = 1 while cation i and partner j are in contact.
  - Intermittent correlation C_I(t) = ⟨h(0)h(t)⟩/⟨h⟩ (contacts that break and reform count).
  - Continuous survival C_C(t) = the fraction of contact time that stays unbroken for at least
    t, after gaps up to t* are filled (Impey, Madden & McDonald, J. Phys. Chem. 87, 5071 (1983)).
    Contacts cut by the trajectory's ends bias C_C low at long t.
  - τ is where C falls to 1/e (linear interpolation). If C has not fallen to 1/e within the
    longest lag, τ is reported as a lower bound.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .uncertainty import blocks, mean_sem


def contacts(positions: np.ndarray, boxes: np.ndarray, a: np.ndarray, b: np.ndarray,
             r_cut_nm: float) -> np.ndarray:  # fmt: skip
    """(frames, len(a), len(b)) bool: |r_a − r_b| < r_cut_nm in the minimum image."""
    out = np.empty((len(positions), len(a), len(b)), dtype=bool)
    for f, (x, h) in enumerate(zip(positions, boxes, strict=True)):
        d = x[a][:, None, :] - x[b][None, :, :]
        s = d @ np.linalg.inv(h)
        s -= np.round(s)
        out[f] = np.linalg.norm(s @ h, axis=-1) < r_cut_nm
    return out


def _components(n_a: int, n_b: int, edges: np.ndarray) -> list[tuple[int, int]]:
    """Connected components of the bipartite graph a ↔ b as (n_a, n_b) per component."""
    parent = list(range(n_a + n_b))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(edges), strict=True):
        ri, rj = find(int(i)), find(n_a + int(j))
        if ri != rj:
            parent[ri] = rj
    counts: dict[int, list[int]] = {}
    for k in range(n_a + n_b):
        counts.setdefault(find(k), [0, 0])[0 if k < n_a else 1] += 1
    return [(p, m) for p, m in counts.values()]


def clusters(contact: np.ndarray, z_cation: int = 1, z_anion: int = -1) -> dict:
    """Cluster populations from (frames, cations, anions) contacts.

    Returns per composition "n₊:n₋" the mean number of clusters per frame and the fractions of
    all cations and of all anions in such clusters, plus the fractions of ions in charged
    clusters (free ions included: they carry current) and of free ions.
    """
    frames, n_p, n_m = contact.shape
    tally: Counter[tuple[int, int]] = Counter()
    for f in range(frames):
        tally.update(_components(n_p, n_m, contact[f]))
    comp = {}
    charged_p = charged_m = 0.0
    for (p, m), k in sorted(tally.items()):
        comp[f"{p}:{m}"] = {
            "clusters_per_frame": k / frames,
            "cation_fraction": k * p / (frames * n_p),
            "anion_fraction": k * m / (frames * n_m),
            "charge_e": p * z_cation + m * z_anion,
        }
        if p * z_cation + m * z_anion != 0:
            charged_p += k * p / (frames * n_p)
            charged_m += k * m / (frames * n_m)
    return {
        "compositions": comp,
        "cations_in_charged_clusters": charged_p,
        "anions_in_charged_clusters": charged_m,
        "free_cation_fraction": comp.get("1:0", {}).get("cation_fraction", 0.0),
        "free_anion_fraction": comp.get("0:1", {}).get("anion_fraction", 0.0),
    }


def shell_composition(per_group: dict[str, np.ndarray], top: int = 12) -> dict:
    """Populations of first-shell combinations: per_group maps a name to (frames, cations,
    partners) contacts. Returns the mean contacts per cation for each group and the `top` most
    frequent combinations with their fractions (of cation-frames)."""
    counts = {name: c.sum(axis=2) for name, c in per_group.items()}  # (frames, cations)
    names = list(counts)
    stacked = np.stack([counts[n] for n in names], axis=-1).reshape(-1, len(names))
    keys, freq = np.unique(stacked, axis=0, return_counts=True)
    order = np.argsort(-freq)[:top]
    combos = {
        " ".join(f"{n}={int(k)}" for n, k in zip(names, keys[i], strict=True)): float(
            freq[i] / len(stacked)
        )
        for i in order
    }
    return {
        "mean_contacts": {n: float(counts[n].mean()) for n in names},
        "combinations": combos,
    }


def _fill_gaps(h: np.ndarray, max_gap: int) -> np.ndarray:
    """Close interior runs of zeros up to max_gap frames long in each column of h (frames, n)."""
    if max_gap <= 0:
        return h
    h = h.copy()
    for col in range(h.shape[1]):
        x = h[:, col]
        on = np.flatnonzero(x)
        if len(on) < 2:
            continue
        gaps = np.diff(on) - 1
        for start, g in zip(on[:-1][(gaps > 0) & (gaps <= max_gap)],
                            gaps[(gaps > 0) & (gaps <= max_gap)], strict=True):  # fmt: skip
            x[start + 1 : start + 1 + g] = True
    return h


def _run_lengths(h: np.ndarray) -> np.ndarray:
    """Lengths of all runs of ones in the columns of h (frames, n)."""
    pad = np.zeros((1, h.shape[1]), dtype=np.int8)
    d = np.diff(np.vstack([pad, h.astype(np.int8), pad]), axis=0)
    starts = np.argwhere(d.T == 1)
    ends = np.argwhere(d.T == -1)
    return ends[:, 1] - starts[:, 1]


def _tau(lags_ps: np.ndarray, c: np.ndarray) -> tuple[float, bool]:
    """(time where c falls to 1/e, True) or (the longest lag, False) if it never does."""
    below = np.flatnonzero(c < 1 / np.e)
    if len(below) == 0:
        return float(lags_ps[-1]), False
    k = below[0]
    if k == 0:
        return float(lags_ps[0]), True
    t0, t1, c0, c1 = lags_ps[k - 1], lags_ps[k], c[k - 1], c[k]
    return float(t0 + (c0 - 1 / np.e) * (t1 - t0) / (c0 - c1)), True


def residence(contact: np.ndarray, dt_ps: float, *, max_lag_fraction: float = 0.5,
              tstar_ps: float = 0.0, chunk: int = 2000) -> dict:  # fmt: skip
    """Intermittent and continuous contact correlations from (frames, a, b) contacts."""
    frames = contact.shape[0]
    h = contact.reshape(frames, -1)
    h = h[:, h.any(axis=0)]  # only pairs that ever touch contribute
    n_lag = max(2, int(max_lag_fraction * frames))
    lags = np.arange(n_lag) * dt_ps
    if h.shape[1] == 0:
        raise ValueError("no contacts within r_cut in this trajectory")
    # intermittent: Σ_pairs Σ_t0 h(t0) h(t0+m) / (frames − m), by FFT in chunks of pairs
    acf = np.zeros(frames)
    for k in range(0, h.shape[1], chunk):
        x = h[:, k : k + chunk].astype(float)
        f = np.fft.rfft(x, n=2 * frames, axis=0)
        acf += np.fft.irfft((f * f.conj()).real, n=2 * frames, axis=0)[:frames].sum(axis=1)
    acf /= frames - np.arange(frames)
    c_int = acf[:n_lag] / acf[0]
    # continuous: a run of L frames gives L − m origins that stay bound for m more frames
    # Σ_L>m (L − m) from suffix sums of the run-length histogram
    runs = _run_lengths(_fill_gaps(h, round(tstar_ps / dt_ps)))
    hist = np.bincount(runs, minlength=frames + 1).astype(float)
    above_n = np.cumsum(hist[::-1])[::-1]  # Σ_{L ≥ m} hist
    above_l = np.cumsum((hist * np.arange(len(hist)))[::-1])[::-1]
    m = np.arange(n_lag)
    c_cont = (above_l[m + 1] - m * above_n[m + 1]) / above_l[1]
    tau_i, ok_i = _tau(lags, c_int)
    tau_c, ok_c = _tau(lags, c_cont)
    return {
        "tau_intermittent_ps": tau_i,
        "tau_intermittent_decayed": ok_i,
        "tau_continuous_ps": tau_c,
        "tau_continuous_decayed": ok_c,
        "tstar_ps": tstar_ps,
        "mean_contact_fraction": float(h.mean()),
        "pairs_with_contact": int(h.shape[1]),
        "lags_ps": lags.round(6).tolist(),
        "C_intermittent": c_int.round(6).tolist(),
        "C_continuous": c_cont.round(6).tolist(),
    }


def with_blocks(contact: np.ndarray, dt_ps: float, n_blocks: int = 5, *,
                tstar_ps: float = 0.0) -> dict:  # fmt: skip
    """`residence` on the whole series (curves), and τ as mean ± sem over blocks.

    In a block the longest lag is half the block, so a τ that is long against the block is a
    lower bound there; `blocks_decayed` counts the blocks where C reached 1/e.
    """
    out = residence(contact, dt_ps, tstar_ps=tstar_ps)
    for key in ("intermittent", "continuous"):
        taus, ok = [], 0
        for s in blocks(contact.shape[0], n_blocks):
            try:
                r = residence(contact[s], dt_ps, tstar_ps=tstar_ps)
            except ValueError:
                continue
            taus.append(r[f"tau_{key}_ps"])
            ok += r[f"tau_{key}_decayed"]
        mean, sem = mean_sem(taus)
        out[f"tau_{key}_blocks"] = {"mean": mean, "sem": sem, "blocks": len(taus),
                                    "blocks_decayed": ok}  # fmt: skip
    return out


def clusters_with_blocks(contact: np.ndarray, n_blocks: int = 5, **kw) -> dict:
    """`clusters` on the whole series, with the block mean ± sem of every fraction."""
    out = clusters(contact, **kw)
    per = [clusters(contact[s], **kw) for s in blocks(contact.shape[0], n_blocks)]
    for key in ("cations_in_charged_clusters", "anions_in_charged_clusters",
                "free_cation_fraction", "free_anion_fraction"):  # fmt: skip
        out[key] = dict(zip(("mean", "sem"), mean_sem([p[key] for p in per]), strict=True))
    for name, comp in out["compositions"].items():
        for key in ("cation_fraction", "anion_fraction"):
            vals = [p["compositions"].get(name, {}).get(key, 0.0) for p in per]
            comp[key] = dict(zip(("mean", "sem"), mean_sem(vals), strict=True))
    out["blocks"] = n_blocks
    return out
