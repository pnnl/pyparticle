# PyParticle

Branch-aware CI and coverage

- CI (pytest): ![CI](https://github.com/lfierce2/PyParticle/actions/workflows/ci-codecov.yml/badge.svg?branch=tests_scaffold)
- Coverage (Codecov): [![codecov](https://codecov.io/gh/lfierce2/PyParticle/branch/tests_scaffold/graph/badge.svg)](https://codecov.io/gh/lfierce2/PyParticle/branch/tests_scaffold)

Notes:
- The badges above show status for the current working branch `tests_scaffold`. Replace the `branch=` query with any branch name to get per-branch stats, e.g. `main`, `refactor_removeBNN`.
- CI runs on all branches (configured in `.github/workflows/ci-codecov.yml`).

Overview

PyParticle provides particle, species, population, and optics modeling utilities.

Key modules (see per-module READMEs):
- `src/PyParticle/` — core package overview
- `src/PyParticle/optics/` — optical particle models and builders
- `src/PyParticle/population/` — population classes and builders
- `src/PyParticle/species/` — species definitions and registry
- `examples/` — runnable examples using JSON configs

Quick start

1) Install (editable):

```bash
pip install -e .
```

Install from GitHub (public):

```bash
pip install git+https://github.com/lfierce2/PyParticle.git
```

Install from GitHub (private) using a personal access token (replace placeholders):

```bash
pip install git+https://<USERNAME>:<TOKEN>@github.com/lfierce2/PyParticle.git
```

Install from GitHub using SSH (requires SSH key access):

```bash
pip install git+ssh://git@github.com/lfierce2/PyParticle.git
```

Notes on private installs:
- Use a GitHub Personal Access Token with `repo` scope for private repositories.
- Avoid embedding long-lived tokens in scripts. Use environment variables or deploy keys when possible.

2) Run tests with coverage:

```bash
pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term
```

3) Examples (optional, gated by env):

```bash
PYPARTICLE_RUN_EXAMPLES=1 python examples/homogeneous_binned_lognormal.py
```

Data

Species property lookups expect data under `datasets/species_data/` (e.g., `aero_data.dat`). See `src/PyParticle/species/base.py`.

Contributing

- Open PRs from feature branches; CI and Codecov badges can be made branch-specific by appending `?branch=<your-branch>` to the badge URLs.
- See `tests/README.md` for the deterministic test scaffold.
  
Note about integration tests: the repository provides a separate `tests/README.md` that documents two test modes (fast default tests and full integration tests). By default CI and the quick test command above run the fast tests only. Integration tests that exercise third-party parcel-model/solver stacks (for example, `pyrcel`'s `ParcelModel.run` which may require Assimulo/CVODE) are intentionally gated behind the developer environment; they are not required for normal development or CI runs unless you opt in and provision the developer environment described in `tests/README.md`.
