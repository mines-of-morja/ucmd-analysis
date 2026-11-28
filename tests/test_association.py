"""D4/D5 on cases with known answers: hand-made graphs, the minimum image, and a two-state
Markov chain whose contact correlations are known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from ucmd_analysis.association import (
    clusters,
    clusters_with_blocks,
    contacts,
    residence,
    shell_composition,
    with_blocks,
)


def test_contacts_use_the_minimum_image() -> None:
    x = np.array([[[0.05, 1.0, 1.0], [1.95, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    box = np.eye(3)[None] * 2.0
    c = contacts(x, box, np.array([0]), np.array([1, 2]), 0.2)
    assert c.tolist() == [[[True, False]]]  # 0.1 nm through the boundary; 0.95 nm direct


def test_cluster_compositions_of_a_known_graph() -> None:
    # cations 0,1 share anion 0 (triple ion +1); cation 2 with anions 1,2 (−1); cation 3 free;
    # anion 3 free; cation 4 with anion 4 (neutral pair)
    edges = np.zeros((1, 5, 5), dtype=bool)
    for i, j in ((0, 0), (1, 0), (2, 1), (2, 2), (4, 4)):
        edges[0, i, j] = True
    out = clusters(edges)
    comp = out["compositions"]
    assert set(comp) == {"0:1", "1:0", "1:1", "1:2", "2:1"}
    assert comp["2:1"]["charge_e"] == 1 and comp["1:2"]["charge_e"] == -1
    assert comp["2:1"]["cation_fraction"] == pytest.approx(0.4)
    assert out["free_cation_fraction"] == pytest.approx(0.2)
    assert out["cations_in_charged_clusters"] == pytest.approx(0.8)  # 2:1, 1:2, free
    assert out["anions_in_charged_clusters"] == pytest.approx(0.8)
    blocked = clusters_with_blocks(np.repeat(edges, 10, axis=0), 5)
    assert blocked["free_cation_fraction"] == {"mean": pytest.approx(0.2), "sem": 0.0}


def test_shell_composition_counts_partners_per_cation() -> None:
    a = np.zeros((2, 2, 3), dtype=bool)
    a[:, 0, :2] = True  # cation 0: two of group A in both frames
    b = np.zeros((2, 2, 1), dtype=bool)
    b[0, 1, 0] = True  # cation 1: one B in frame 0
    out = shell_composition({"A": a, "B": b})
    assert out["mean_contacts"] == {"A": 1.0, "B": 0.25}
    assert out["combinations"] == {"A=0 B=0": 0.25, "A=0 B=1": 0.25, "A=2 B=0": 0.5}


def markov(q: float, r: float, frames: int, pairs: int, seed: int) -> np.ndarray:
    """Per frame: a contact breaks with probability q, a free pair binds with probability r."""
    rng = np.random.default_rng(seed)
    h = np.empty((frames, pairs), dtype=bool)
    h[0] = rng.random(pairs) < r / (q + r)
    u = rng.random((frames, pairs))
    for f in range(1, frames):
        h[f] = np.where(h[f - 1], u[f] >= q, u[f] < r)
    return h[:, :, None]


def test_markov_contacts_give_the_exact_correlations() -> None:
    q, r, dt = 0.05, 0.01, 2.0
    h = markov(q, r, 20000, 40, seed=5)
    out = residence(h, dt, max_lag_fraction=0.01)
    m = np.arange(len(out["lags_ps"]))
    # continuous: geometric run lengths → C_C(m) = (1 − q)^m, τ = −dt/ln(1 − q) = 39.0 ps
    assert np.allclose(out["C_continuous"], (1 - q) ** m, atol=0.01)
    assert out["tau_continuous_ps"] == pytest.approx(-dt / np.log(1 - q), rel=0.03)
    # intermittent: C_I(m) = p + (1 − p)(1 − q − r)^m with p = r/(q + r)
    p = r / (q + r)
    assert np.allclose(out["C_intermittent"], p + (1 - p) * (1 - q - r) ** m, atol=0.01)
    tau_i = dt * np.log((1 / np.e - p) / (1 - p)) / np.log(1 - q - r)
    assert out["tau_intermittent_ps"] == pytest.approx(tau_i, rel=0.03)
    # filling one-frame gaps joins runs: C_C decays more slowly
    filled = residence(h, dt, max_lag_fraction=0.01, tstar_ps=dt)
    assert filled["tau_continuous_ps"] > out["tau_continuous_ps"]
    blocked = with_blocks(h, dt, 5)
    tb = blocked["tau_continuous_blocks"]
    assert tb["blocks_decayed"] == 5
    assert tb["mean"] == pytest.approx(-dt / np.log(1 - q), abs=4 * tb["sem"] + 1.0)


def test_no_contact_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="no contacts"):
        residence(np.zeros((10, 2, 2), dtype=bool), 1.0)


def test_cli_runs_the_three_analyses(tmp_path, capsys) -> None:
    import json

    from test_a1_core import spec_doc, write_h5md

    from ucmd_analysis.cli import main

    rng = np.random.default_rng(2)
    n, frames, box = 20, 50, 2.0
    x = (
        np.cumsum(rng.normal(0, 0.05, (frames, 2 * n, 3)), axis=0)
        + rng.random(2 * n * 3).reshape(2 * n, 3) * box
    )
    write_h5md(tmp_path / "t.h5", spec_doc(n, box), np.arange(frames) * 1.0, x, box)
    for args in (["clusters", "--cation", "Li:Li", "--anion", "A:F", "--r-cut-nm", "0.4"],
                 ["shell", "--a", "Li:Li", "--b", "A:F", "--r-cut-nm", "0.4"],
                 ["residence", "--a", "Li:Li", "--b", "A:F", "--r-cut-nm", "0.4"]):  # fmt: skip
        assert main([args[0], str(tmp_path / "t.h5"), *args[1:]]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["frames"] == frames
