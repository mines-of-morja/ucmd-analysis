# ucmd-analysis

The trajectory analysis of **UnitcellMD**. It reads system packages and H5MD trajectories and
computes structure (RDFs, coordination, ion association), self-diffusion, and Onsager transport
coefficients, giving conductivity and transference numbers, each with an uncertainty. It does
not depend on the engine that produced the trajectory.

**Status:** A1 core on branch `a1`. Plan and shared spec:
[UnitcellMD](https://github.com/mines-of-morja/UnitcellMD).

| Command | What it gives |
|---|---|
| `ucmd-analyze transport TRAJ.h5 --temperature-K T [--viscosity-mPa-s η]` | D of every species (Yeh–Hummer-corrected with η); Onsager L₊₊, L₋₋, L₊₋; κ, κ_NE, ionicity, t₊ (COM frame); κ split into self and distinct parts |
| `ucmd-analyze rdf TRAJ.h5 --a Li:Li --b EC:O:1` | g(r), first peak, coordination number |
| `ucmd-analyze density TRAJ.h5` | density |
| `ucmd-analyze clusters TRAJ.h5` | Li⁺–PF₆⁻ cluster populations (free, pairs, triple ions, …) |
| `ucmd-analyze shell TRAJ.h5 --b EC:O:1 --b EMC:O:1 --b PF6:P` | first-shell composition of Li⁺ |
| `ucmd-analyze residence TRAJ.h5 --b EC:O:1` | intermittent and continuous residence times |

Every number carries a block error (`--blocks`, default 5). Contact cutoffs default to the first
minimum of g(r) and are reported. Atom groups are `MOLECULE:ELEMENT[:BONDS]`, e.g. `EC:O:1` for
the carbonyl oxygen. Output is JSON.

**Planned** (PLAN.md, track A): Green–Kubo Onsager coefficients, reference-frame
transformations (solvent and volume frames).

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
