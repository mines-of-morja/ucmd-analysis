# ucmd-analysis

The trajectory analysis of **UnitcellMD**. It reads system packages and H5MD trajectories and
computes structure (RDFs, coordination, ion association), self-diffusion, and Onsager transport
coefficients, giving conductivity and transference numbers, each with an uncertainty. It does
not depend on the engine that produced the trajectory.

**Status:** Phase 0 skeleton. Package layout and checks only.

**Planned** (PLAN.md, track A):

- readers;
- molecular centres of mass and the barycentric frame;
- RDF, MSD with finite-size correction;
- Onsager L_ij in Einstein and Green–Kubo form, reference-frame transformations;
- block and replica statistics;
- then the reproduction of Lehnert et al. 2025 (J. Electrochem. Soc. 172, 050523).

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Python ≥ 3.12.

## License

MIT; see [LICENSE](LICENSE).
