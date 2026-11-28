# A1.7 target 1: S⁰_AB of Cheng's binary LJ mixture — result

Plan and criterion: `README.md` (committed at 6d072a5, before the runs). Output: `result.json` (a copy of
`out/result.json`, written by `analyze.py`).

## Verdict: pass

| | value | criterion |
|---|---|---|
| S⁰_AB, NPT, scalar form, k ≤ 2π · 0.2/σ, 5001 frames | **−0.4373 ± 0.0017** | |
| Cheng 2022, N = 4000 | −0.439 ± 0.004 | |
| difference | +0.0017 | ≤ 2 √(SE² + 0.004²) = 0.0087 → **pass** |
| condition: block / slowest relaxation time | 2151 ps / 64.5 ps = 33 | ≥ 5, met |

± is one standard error, a jackknife over 10 blocks.

## Reported, not used for the verdict

| | NPT | NVT (mean NPT volume) |
|---|---|---|
| S⁰_AB, scalar, k ≤ 2π · 0.15/σ | −0.4409 ± 0.0052 | −0.4373 ± 0.0076 |
| S⁰_AB, scalar, 0.2 (the criterion's) | −0.4373 ± 0.0017 | −0.4363 ± 0.0043 |
| S⁰_AB, scalar, 0.25 | −0.4372 ± 0.0015 | −0.4367 ± 0.0024 |
| S⁰_AB, scalar, 0.3 | −0.4378 ± 0.0013 | −0.4371 ± 0.0014 |
| S⁰_AB, matrix, 0.2 | −0.4371 ± 0.0017 | −0.4360 ± 0.0043 |
| ∂(βμ_A)/∂ ln x_A, matrix, 0.2 (ideal: 1) | 0.998 ± 0.003 | 1.001 ± 0.009 |
| ρ k_B T κ_T from S⁰ (matrix, 0.2) | 0.1277 ± 0.0004 | 0.1268 ± 0.0004 |
| ρ k_B T κ_T from the volume fluctuations | 0.1264 ± 0.0021 | — |
| reduced density ρσ³ | 0.4954 | 0.4955 |

- **NVT against NPT:** S⁰_AB differs by 0.23 combined SE. At this box size the NVT run gives the same
  limit, so the NVT transport pilot's frames can be used for the method, as design §6 assumed.
- **Compressibility:** the value from S⁰ and the one from the volume fluctuations of the same run differ
  by 0.6 combined SE. The second does not use the k → 0 fit, so this checks the fit and the normalisation
  independently of Cheng's number.
- **Fit range:** from 0.2 to 0.3 · 2π/σ the value moves by less than 1 SE; at 0.15 (fewer shells) it is
  noisier and 0.7 SE lower.
- **The non-ideal part is near zero,** as the README predicted: ∂(βμ_A)/∂ ln x_A = 0.998 ± 0.003, and
  (ρ k_B T κ_T − 1)/2 = −0.436 alone accounts for S⁰_AB. **This target tests the normalisation, the scaled
  coordinates, the k → 0 fit and the sampling, not the thermodynamic factor of a non-ideal mixture.** The
  NaCl target (design §4.2) is the test of that. No electrolyte TDF is reported on this result alone.

## Where and how it ran

- **Machine:** soricel-pc (AMD Ryzen 5 4500, WSL2), 4 threads, `nice -n 10`, run by that computer's agent
  session from the hand-over of 2026-11-26 (author's approval to move it there). Its GPU ran other work during
  the NPT stage.
- **Versions:** engine `ucmd 0.0.0 (b9f161ebba4a)` (branch `s012-pair-overrides`, spec 0.1.2 pair
  overrides), printed by every stage; ucmd-analysis 6d072a5, `validation/a17-lj` unchanged.
- **Command:** `A17_UCMD=… A17_PYTHON=… UCMD_MACHINE_PROFILE=… nice -n 10 bash validation/a17-lj/run.sh`;
  tests before the run: `PYTHONPATH=src pytest validation/a17-lj` → 3 passed.
- **Wall time:** equilibration 273 s (5·10⁵ steps), NPT 5603 s, NVT 4588 s (10⁷ steps each), 331.8 and
  405.1 ns/day. Started 2026-11-26 09:51 UTC, `run.sh` exit 0 at 11:53:45 UTC. Nothing was stopped or
  restarted.
- **Kept on soricel-pc:** the two trajectories (243 MB each). Copied back: `result.json`, `analyze.log`,
  `out-run.log` and the three thermo tables.
- **Mean NPT volume** 317.3076 nm³, NVT box scaled by 1.000684.
