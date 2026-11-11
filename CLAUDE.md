# ucmd-analysis — agent notes

The shared rules are in `../CLAUDE.md`, in the UnitcellMD umbrella repo one level up in the
workspace. The plan is `../PLAN.md`, track A (§7).

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # once
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Repo rules

- Reads system packages and H5MD trajectories only. Never imports builder, engine, bridge or
  launcher code; `tests/test_architecture.py` enforces this.
- Every reported transport number carries an uncertainty (block averages or replicas).
- Validate each method on synthetic data with a known answer before using real trajectories.
