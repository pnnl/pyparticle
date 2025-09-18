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

---

Edit this plan as needed. When you're ready, give me permission to apply the first archival commits and run tests.
