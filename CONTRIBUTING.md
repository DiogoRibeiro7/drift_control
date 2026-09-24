# Contributing to drift_control

## Getting set up

```bash
git clone https://github.com/DiogoRibeiro7/drift_control.git
cd drift_control
pip install poetry
poetry install --with dev --extras "viz concept ml polars stream mlflow"
```

`apache-airflow` is deliberately left out: the Airflow wrapper is tested
through a faked module, and airflow's pins make a clean resolve slow and
brittle.

## The development loop

```bash
poetry run ruff check .
poetry run mypy
poetry run pytest -q --cov
```

If those three pass, CI should pass. Dev tool versions come from
`poetry.lock`, so everyone runs the same ones; if you change them, commit the
updated lock.

## Code style

- `ruff` for linting and import order, enforced in CI.
- Type annotations on public APIs. `mypy` checks the **whole** package, not a
  subset, and the package ships `py.typed` — annotations are part of the API.
- Prefer fixing a type error over silencing it. If an ignore is genuinely
  right, comment why.

## Tests

- Tests live in `tests/` and run with `pytest`.
- Coverage has a floor of 85% (`fail_under` in `pyproject.toml`). It is a
  regression guard, not a target to game.
- A test that needs an optional extra should skip cleanly when it is absent —
  the `core-only` CI job installs without extras.
- When you fix a bug, add the test that fails before the fix.

## Pull requests

1. Branch from `main`, keep the branch short-lived.
2. Keep PRs focused, with a clear title and a linked issue where one exists.
3. Add or update tests for the behaviour you change.
4. Note user-visible changes in `CHANGELOG.md`.

## Architecture

`ROADMAP.md` describes the target package layout and the phase plan.
`TECHNICAL_DEBT_ROADMAP.md` tracks what is known to be outstanding — read it
before starting anything substantial, and update it when you close something
out.
