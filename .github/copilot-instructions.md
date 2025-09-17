## Quick orientation for AI coding agents (concise)

This repo implements PyParticle: aerosol particle models, population builders,
and optics. The package is under `src/PyParticle` (src-layout). Tests and
examples drive most workflows and use JSON/YAML configs in `examples/configs/`.

Key files to inspect first
- `src/PyParticle/__init__.py` — public exports and dataset path handling
- `src/PyParticle/population/builder.py` — `build_population(cfg)` API
- `src/PyParticle/optics/builder.py` — optics construction and required grids
- `src/PyParticle/viz/` — plotting helpers; layout (`layout.py`), plotting
  (`plotting.py`), formatting (`formatting.py`) and grid helpers
  (`grids.py`).

Developer workflows (exact commands)
- Install editable: `pip install -e .`
- Run fast tests: `pytest -q --maxfail=1 --disable-warnings`
- Reference comparison harness (optional):
  `pytest -q tests/run_all_comparisons.py --input examples/configs/ccn_single_na_cl.yml --compare both --output reports/reference_report.json`

Project-specific flags used by tests
- `PYPARTICLE_DATA_PATH` — override bundled `datasets/` resolution
- `PYPARTICLE_RUN_EXAMPLES=1` — enable examples in test runs
- `PYTEST_ALLOW_NETWORK=1` — allow network during tests (default blocked)

Viz & plotting conventions (important)
- `plot_lines(varname, (pop,), var_cfg, ax)` returns `(Line2D, labs)` and
  does not modify axis labels/titles; callers call `format_axes` + `add_legend`.
- Wavelength grids for optics are in meters. Examples accept `wvl_grid_um` and
  convert to meters (`*1e-6`) before building optics.
- Layout: use `viz.layout.make_grid(rows, cols)` to get (fig, axes).
- New grid helpers provided in `viz.grids`:
  - `make_grid_popvars(rows, columns, ...)` — Type A: rows are populations
    (or config dicts), columns are variable names. Each cell plots one var.
  - `make_grid_scenarios_timesteps(rows, columns, variables, ...)` — Type B:
    rows are scenario config dicts, columns are timesteps; multiple variables
    may be plotted on each axis.
  - `make_grid_mixed(rows, columns, ...)` — Type C: rows may mix prebuilt
    populations and config dicts; columns are variable names.
  These helpers build populations when a config dict is provided (calls
  `build_population`) and call `plot_lines`, `format_axes`, `add_legend`.

Patterns to follow when editing code
- Keep public API stable in `src/PyParticle/__init__.py` — examples/tests
  import from the package root.
- When changing species/data: update `datasets/species_data/` and adjust
  `PYPARTICLE_DATA_PATH` in tests/CI if needed.

Where to add tests and small PRs
- Add unit tests under `tests/unit/`. Integration or long-running reference
  comparisons go under `tests/integration/` or use the reference harness with
  adapters in `tests/reference_wrappers/`.

If anything is missing or you want this expanded (debugging tips, example
walkthrough), tell me which area and I'll extend this file.
