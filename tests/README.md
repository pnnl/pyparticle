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