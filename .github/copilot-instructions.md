# Instructions for coding agents (GitHub Copilot and others)

This repo is one component of UnitcellMD. Its rules: `CLAUDE.md` here, and the shared rules in
`../CLAUDE.md` and `../CONTRIBUTING.md` of the UnitcellMD umbrella repo, which this repo is
cloned into (https://github.com/mines-of-morja/UnitcellMD, branch `b2a`). Read them at the start
of every session; they are binding. The essentials:

- **Branches:** never commit to `main` or `a1`. Work on `<name>/<task-id>-<short-title>` made
  from `a1`, and open a pull request into `a1`.
- **Boundaries:** reads system packages and H5MD trajectories only; never imports builder, engine
  or bridge code (`tests/test_architecture.py`).
- **Units:** nm, ps, amu, kJ/mol, e, K; constants only from `src/ucmd_analysis/constants.py`.
- **Test-first:** every method is validated on synthetic data with a known answer, then against
  a published number, before it touches real trajectories. Every reported transport number has
  an uncertainty. Never loosen a tolerance, skip or delete a failing test.
- **Design before code:** a new method starts as a short design note in a draft pull request;
  implement after the author approves it. Never invent a physical parameter.
- Checks: `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`.
