# ucmd-analysis

The trajectory analysis of **UnitcellMD**. It reads system packages and H5MD trajectories and
computes structure (RDFs, coordination, ion association), self-diffusion, and Onsager transport
coefficients, giving conductivity and transference numbers, each with an uncertainty. It does
not depend on the engine that produced the trajectory.

**Status:** A1 core on branch `a1`. Plan and shared spec:
[UnitcellMD](https://github.com/mines-of-morja/UnitcellMD).

| Command | What it gives |
|---|---|
| `ucmd-analyze transport TRAJ.h5 --temperature-K T [--viscosity-mPa-s η] [--nu-e-cm3-per-mol ν_e]` | D of every species (Yeh–Hummer-corrected with η); Onsager L₊₊, L₋₋, L₊₋; κ, κ_NE, ionicity; t₊ in the COM and SOL frames and, with the salt's partial molar volume ν_e, in the VOL frame (conventions A and B, ADR 0013); κ split into self and distinct parts |
| `ucmd-analyze diffusion TRAJ.h5 --temperature-K T [--species A,B] [--viscosity-mPa-s η]` | D of neutral species such as pure water (block errors; Yeh–Hummer-corrected with η, cubic boxes). `transport` refuses neutral ions and points here |
| `ucmd-analyze onsager-gk TRAJ.h5 --temperature-K T --plateau-ps LO HI --max-lag-ps M` | L_ij, κ, t₊ (COM) from velocities (Green–Kubo) next to the Einstein form over the same lags: a check of the analysis, not an independent estimate. Needs H5MD velocities at a fine stride |
| `ucmd-analyze rdf TRAJ.h5 --a Li:Li --b EC:O:1` | g(r), first peak, coordination number |
| `ucmd-analyze density TRAJ.h5` | density |
| `ucmd-analyze clusters TRAJ.h5` | Li⁺–PF₆⁻ cluster populations (free, pairs, triple ions, …) |
| `ucmd-analyze shell TRAJ.h5 --b EC:O:1 --b EMC:O:1 --b PF6:P` | first-shell composition of Li⁺ |
| `ucmd-analyze residence TRAJ.h5 --b EC:O:1` | intermittent and continuous residence times |
| `ucmd-analyze dielectric TRAJ.h5 --temperature-K 298.15` | static ε from the total dipole of the neutral molecules (tin-foil boundaries) |
| `ucmd-analyze hvap --liquid L.tsv --molecules N --gas G.tsv --temperature-K T` | ΔH_vap = ⟨U_gas⟩ − ⟨U_liq⟩/N + RT from two thermo tables |
| `ucmd-analyze viscosity THERMO.tsv --temperature-K T` | Green–Kubo shear viscosity from the pressure-tensor columns (spec 0.1.1), block errors and the running integral |

Every number carries a block error (`--blocks`, default 5). Contact cutoffs default to the first
minimum of g(r) and are reported. Atom groups are `MOLECULE:ELEMENT[:BONDS]`, e.g. `EC:O:1` for
the carbonyl oxygen. Output is JSON.

**Planned** (PLAN.md, track A): per-solvent frames of t₊, thermodynamic factor.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Python ≥ 3.12.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
