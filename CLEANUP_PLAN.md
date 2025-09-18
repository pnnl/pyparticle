# PyParticle — Cleanup Plan (editable)

This file describes a safe, reversible plan for cleaning up clutter and legacy
files in the repository. It's written to be followed step-by-step and is
intentionally conservative: we archive rather than delete on first pass.

## Goals
- Remove generated artifacts and caches from source control.
- Move obvious legacy/archived code to `archive/` so it's out of main tree but still available.
- Keep the repository test-green at every step.

## Safety rules
1. Always create a branch for the cleanup work: `git checkout -b housekeeping/cleanup-clutter`.
2. For each candidate file, search for references before moving: `git grep -n "<filename>"`.
3. Move rather than delete on first pass: `mkdir -p archive/<area>` then `git mv <file> archive/<area>/`.
4. Run unit tests & smoke tests after each logical group of changes.
5. If tests fail, revert the last commit and investigate; do not continue until resolved.

## First-pass candidates (low-risk)
- Remove generated metadata and caches (delete from repo):
  - `pyparticle.egg-info/` (packaged metadata)
  - `src/PyParticle/__pycache__/` and other `__pycache__` directories
  - Any top-level generated images that are not canonical (e.g., `examples/*.png`) — review each before removal.

## Medium-risk candidates (archive-first)
- `src/PyParticle/analysis_archive.py` -> `archive/pyparticle-legacy/`
- `tests_archive/` -> `archive/tests_archive/`
- `tools/` scripts that are not used in CI or developer workflows: inspect and archive selectively.
- `examples/out_grid_partmc_mam4.png` -> `archive/examples_assets/`

## High-risk candidates (manual review)
- Anything under `population/factory/partmc.py` or `mam4.py`: these rely on external data and heavy deps (netCDF4). Do not remove; only refactor if you have a replacement.
- Modules referenced by `discover_*` registries: ensure tests or examples do not dynamically import them.

## Commands (zsh)
- create branch:
```bash
git checkout -b housekeeping/cleanup-clutter
```
- search references for a candidate (example):
```bash
git grep -n "analysis_archive" || true
```
- move a file to archive:
```bash
mkdir -p archive/pyparticle-legacy
git mv src/PyParticle/analysis_archive.py archive/pyparticle-legacy/
git commit -m "chore: archive legacy analysis_archive.py"
```
- remove package metadata / caches from repo:
```bash
git rm -r --cached pyparticle.egg-info || true
git rm -r --cached src/PyParticle/__pycache__ || true
git commit -m "chore: remove generated package metadata and caches"
```

## Verification steps
1. Run unit tests:
```bash
conda run -n pyparticle pytest -q --maxfail=1
```
2. Run a minimal smoke example that builds a small population and computes optics (choose a lightweight example from `examples/`).
3. Lint the changed files (optional): `ruff check .` or `flake8`.

## After checks
- If tests pass and smoke checks look good, consider deleting files from `archive/` or keeping them as historical backup for a release branch.

## Notes & follow-ups
- Add `archive/` to the repository README or `OVERVIEW.md` explaining what's stored there and why.
- Add or update `.gitignore` to exclude `__pycache__`, `*.egg-info`, `*.pyc`, and other build artifacts.
- Consider moving large example outputs to a release artifact store or `examples/assets/` and update docs.

## Make `pyparticle` conda environment the default in VS Code

Developers should use the `pyparticle` (or `pyparticle-partmc` for PARTMC/MAM4 work) conda environment. The following steps make that the default workspace behavior in VS Code and ensure integrated terminals activate the environment automatically.

1. Create / verify the conda env(s) locally (if not already):

```bash
# create the dev env from the repo environment file (optional)
conda env create -f environment-dev.yml -n pyparticle

# or ensure the env exists for heavy workflows
conda env create -f environment-partmc.yml -n pyparticle-partmc
```

2. Find the full path to the python executable in the conda env (one of these):

```bash
conda activate pyparticle
which python
# or (non-interactive): conda run -n pyparticle which python
```

3. Recommended: set the workspace interpreter and terminal activation by creating `.vscode/settings.json` in the repo root (workspace settings). Edit the path below to the value from step 2:

```json
{
  "python.defaultInterpreterPath": "/full/path/to/conda/envs/pyparticle/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.condaPath": "/full/path/to/conda/bin/conda"  
}
```

Notes:
- `python.defaultInterpreterPath` points VS Code to the exact python executable in the conda environment. Replace `/full/path/to/...` with the `which python` value from step 2.
- `python.terminal.activateEnvironment: true` instructs the Python extension to activate the selected interpreter for new integrated terminals (so `conda activate pyparticle` runs automatically).
- Optionally set `python.condaPath` to your conda/conda.sh location if VS Code cannot find conda automatically.

4. Alternative (manual): from VS Code Command Palette (⇧⌘P) run `Python: Select Interpreter` and choose the `pyparticle` interpreter; then ensure `Python > Terminal: Activate Environment` is enabled in workspace settings.

5. For PARTMC / MAM4 workflows: repeat steps using `pyparticle-partmc` and a separate workspace folder or alter the `python.defaultInterpreterPath` to point to that env.

This ensures tests and example runs launched from VS Code terminals use the correct conda environment by default.

---

Edit this plan as needed. When you're ready, give me permission to apply the first archival commits and run tests.

## Diagnosing missing imports (recommended flow)

When running tests or examples you may see ImportError for packages that should normally be present. Follow this flow:

1. Confirm you're running inside the `pyparticle` conda env (preferred). In a terminal run:

```bash
conda run -n pyparticle python -c "import <module>; print('<module> OK')" || echo "<module> missing in pyparticle"
```

2. If the module is missing, check whether the test or example requires the heavier PARTMC/MAM4 stack (these require `netCDF4` and other packages). Try the `pyparticle-partmc` env (if it exists):

```bash
conda run -n pyparticle-partmc python -c "import <module>; print('<module> OK')" || echo "<module> missing in pyparticle-partmc"
```

3. If `pyparticle-partmc` does not exist, generate it from the base `environment.yml` using the provided helper script and then create the conda env (see next section).

4. If you intentionally want to use the lighter env, install the missing package into that env (or use `pip install` inside the activated env):

```bash
conda run -n pyparticle conda install <pkg> -c conda-forge -y || conda run -n pyparticle pip install <pkg>
```

Always prefer creating or using the `pyparticle-partmc` env for PARTMC/MAM4 workflows rather than bloating the lightweight env.

## Generating `environment-partmc.yml` (and `environment-dev.yml`)

This repository contains a helper script `tools/generate_env_variants.py` that produces `environment-dev.yml` and `environment-partmc.yml` derived from `environment.yml`.

Run the script from the repo root using a Python that has PyYAML available (the `pyparticle` env already includes PyYAML in most setups):

```bash
conda run -n pyparticle python tools/generate_env_variants.py --write
```

The script will write `environment-dev.yml` and `environment-partmc.yml` next to `environment.yml`. `environment-partmc.yml` includes `netcdf4` and other heavy dependencies used by PARTMC/MAM4 population builders.

After generating, create the env with:

```bash
conda env create -f environment-partmc.yml -n pyparticle-partmc
```

And for the dev environment (optional):

```bash
conda env create -f environment-dev.yml -n pyparticle-dev
```

If PyYAML is not available in the Python running the script, the helper will prompt to install it first (`pip install pyyaml`).

### Automated helper

To automate generation, creation/update, and running tests under the heavier PARTMC env, use the helper script:

```bash
tools/setup_partmc_env.sh
```

Options:
- `--no-create` : generate env files and skip creating/updating the conda env
- `--no-tests`  : generate and create/update env but skip running tests

This script will:
1. Run `tools/generate_env_variants.py --write` (via `pyparticle` env if available)
2. Create or update `pyparticle-partmc` using the generated `environment-partmc.yml`
3. Run unit tests under `pyparticle-partmc`


