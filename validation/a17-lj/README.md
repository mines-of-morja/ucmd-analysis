# A1.7 target 1: S⁰_AB of a binary repulsive LJ mixture (Cheng 2022)

Plan of the first known-answer test of the thermodynamic-factor code on a simulation (design
`docs/design/A1.7-tdf.md` §4.1 of UnitcellMD, approved by the author on 2026-11-25). **Written and committed
before any run.** Results go into `REPORT.md`.

## Target

Cheng 2022 (arXiv 2205.03070v2, p. 3 and Fig. 2a; J. Chem. Phys. 157, 121101), read in the preprint:
6-12 LJ cut and shifted at 2^(1/6) σ, all σ equal, ε_AA = 1.2, ε_BB = 1.0, ε_AB = 1.1, T = 1.2, P = 2,
x_A = 0.5. **S⁰_AB = −0.439 ± 0.004 (N = 4000)**, −0.436 ± 0.005 (23 328), −0.438 ± 0.004 (108 000). What the
± means is not stated; taken as one standard error.

## Protocol

| | Cheng 2022 | here |
|---|---|---|
| N, steps, time step | 4000, 10⁷, 10⁻³ τ | the same |
| thermostat | stochastic velocity rescaling | CSVR (the same), 1 ps |
| barostat | Nosé–Hoover | C-rescale, isotropic, 2 ps, 2·10⁻⁴ /bar |
| frames | every 2000 steps | the same (5000 frames) |
| k range of the fit | k ≤ 2π · 0.2/σ, 1/S_AB linear in k² (their Eq. 18) | the same, weighted by modes |
| coordinates | scaled (their Eq. 17) | the same |

- The scale is Rahman's argon (σ 0.34 nm, ε/k_B 120 K, 39.948 amu); reduced results do not depend on it.
  Masses are not given by Cheng and are taken equal; they do not enter S(k).
- Thermostat and barostat times and the compressibility set relaxation rates, not averages. The
  compressibility is that of `ucmd-engine/validation/npt-argon`.
- ε_AB is a spec 0.1.2 pair override. Engine: ucmd b9f161e (`s012-pair-overrides`).
- Stages (`run.sh`): 5·10⁵ steps NPT from fcc with random labels (discarded); 10⁷ steps NPT;
  10⁷ steps NVT at the mean volume of the NPT run, as design §6 asks.

## Criterion (fixed before the runs)

**NPT run passes if |S⁰_AB + 0.439| ≤ 2 √(SE² + 0.004²)**, with S⁰_AB from all frames, fit form `scalar`,
k ≤ 2π · 0.2/σ, and SE from a jackknife over 10 blocks. Condition: a block is at least 5 relaxation
times of ρ_A at the smallest k. If not, the run is continued, not judged.

Nothing is tuned on the result. A failure is reported as a failure, with the checks below.

## Reported, not used for the verdict

- NVT: S⁰_AB and its difference from NPT in combined SE (is the NVT pilot usable?).
- Fit range 0.15, 0.25, 0.30 · 2π/σ and the form `matrix`.
- Compressibility: ρ k_B T κ_T from S⁰ (matrix form) against ⟨δV²⟩ ρ/⟨V⟩ of the NPT run. Independent of the
  fit, but not of the sampling.
- ∂(βμ_A)/∂ ln x_A from Cheng's Eq. 21, in both fit forms.

## What this target does not test

For two equal species S⁰_AB = (ρ k_B T κ_T − 1)/2. At a compressibility factor near 0.1 that is about
−0.45, so −0.439 is mostly the compressibility. The target tests the normalisation, the scaled
coordinates, the k → 0 fit and NPT sampling. It hardly tests the non-ideal part, which is what a
thermodynamic factor is about; Cheng's Fig. 2c shows excess chemical potentials below 0.1 here. The
NaCl target (design §4.2) is the test of that. No electrolyte TDF is reported after this target alone.

## Cost

Measured 2026-11-25: 6000 steps in 4.5 s (8 threads) or 8.3 s (1 thread). 10⁷ steps: about 2.1 h (8 threads)
or 3.8 h (1 thread); both production runs and the start about 4.5 h on 8 threads.

## Seen before the runs

A dry run of `run.sh` in a copy, with 1/250 of the production steps (101 frames per run, interrupted once
by SIGTERM and continued), to test the scripts. Its S⁰_AB (scalar, 0.2) was −0.36 ± 0.05 and the analysis
refused to judge it, because its blocks are shorter than the relaxation time (25 ps). Nothing in the plan
or the criterion was changed after it.

## Commands

`bash validation/a17-lj/run.sh` (queue job, `cpu`); tests of the analysis:
`PYTHONPATH=src pytest validation/a17-lj`.
