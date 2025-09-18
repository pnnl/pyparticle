## Running tests

This project provides two testing modes so you can run a fast, local-only test set without heavy developer dependencies, or run the full integration tests that exercise third-party reference libraries.

Option 1 — Default (fast) tests (recommended for local development / CI)
- Purpose: run unit tests and lightweight package smoke tests. These do NOT require the developer integration environment and are quick to execute.
- Command (from repository root):

```bash
# run unit tests and a couple of top-level smoke tests
pytest -q tests/unit tests/test_package_smoke.py tests/test_examples_discovery.py
```

Notes:
- This runs the tests that are expected to be portable across most developer machines and CI.
- It intentionally excludes `tests/integration/` which exercise optional, heavier dependencies.

Option 2 — Full integration tests (requires developer environment)
- Purpose: exercise integration tests that call third-party reference libraries (for example, `pyrcel` for CCN/parcel-model tests and `PyMieScatt` for optics tests). These tests can be sensitive to compiled dependencies and environment configuration.
- Two ways to run the integration tests:

1) Using the helper script (recommended):

```bash
# this script prepares the dev environment (if needed) and runs the integration tests
./tools/run_integration.sh
```

2) Manually (if you prefer to create/activate your own environment):

```bash
# create/activate a conda environment suitable for integration tests
# (the helper script may generate an environment-dev.yml for you)
conda env create -f environment-dev.yml -n pyparticle-dev
conda activate pyparticle-dev

# then run the integration tests
pytest -q tests/integration
```

Common issues and troubleshooting
- Some integration tests (historically tests involving `pyrcel`'s `ParcelModel.run`) depend on external integrator backends (for example, Assimulo / CVODE bindings). If you see errors referencing missing classes like `Explicit_Problem` or integrator backends, install the required integrator packages in your dev environment or use the helper script so it sets up a working environment.
- Warnings about binary incompatibility (e.g. numpy C-API size mismatches) are usually environmental; they can often be resolved by creating a clean Conda environment and installing the pinned packages.

Tips
- If you want CI to run only the fast tests, use the command in Option 1.
- If you want CI to run integration tests, ensure the CI image installs the same dev environment as you (the helper script documents expected dependencies).

If you'd like, I can also:
- Add a short section to the repository root `README.md` linking to this file, or
- Update `tools/run_integration.sh` to explicitly create `environment-dev.yml` with the necessary integrator packages (Assimulo) so ParcelModel.run can be exercised in CI.
# Tests for PyParticle

This test suite is designed to be:

- Deterministic: RNGs are seeded; plotting uses a headless backend (Agg).
- Offline-friendly: Network access is disabled by default during tests.
- GPU-free by default: CUDA is disabled via `CUDA_VISIBLE_DEVICES=""`.
- Fast: Unit tests are tiny and examples are gated.

## Markers

- `unit`: fast unit tests.
- `examples`: example discovery and (optionally) fast execution.
- `integration`: slower or external-integration tests (currently placeholders).

Markers are registered in `conftest.py` to avoid warnings.

## Environment variables

- `PYPARTICLE_RUN_EXAMPLES=1` — enable running example `.py` scripts in fast mode.
- `PYPARTICLE_FAST=1` — set by fixtures when running examples to hint "fast" behavior.
- `MPLBACKEND=Agg` — headless plotting (set by fixture).
- `CUDA_VISIBLE_DEVICES=""` — disable GPUs (set by fixture).
- `PYTEST_SEED=1337` — seed for RNGs.

## How to run

- Unit-only:

```
pytest -q -m unit
```

- All tests (examples off by default):

```
pytest -q
```

- Run examples (gated):

```
PYPARTICLE_RUN_EXAMPLES=1 pytest -q -m examples -k run_examples
```

- Coverage:

```
pytest --cov=. --cov-report=term-missing
```

## Contributing

- Add small, deterministic fixtures.
- Keep tests offline and GPU-free by default.
- Mark appropriately with `unit` / `examples` / `integration`.
- Prefer tiny synthetic data over real downloads.