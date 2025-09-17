AI request: refactor analysis -> viz/data_prep and design plot-ready data APIs
=======================================================================

Goal for the other AI
---------------------
Produce a clear, actionable patch (or sequence of patches) that refactors the current
`src/PyParticle/analysis.py` responsibilities into a new module `src/PyParticle/viz/data_prep.py`.
The new module should expose small, well-documented functions that return "plot-ready" data for
each plot type used by the viz layer (line plots and scatter plots). Keep the public plotting
API stable where practical; plotting helpers (viz.plotting, viz.grids) should call the new data
prep functions and only handle plotting/formatting.

Deliverables requested from the other AI
---------------------------------------
1) A precise patch (git-style diff) that creates `src/PyParticle/viz/data_prep.py` and moves/refactors
   the relevant functions from `src/PyParticle/analysis.py` into it.
2) Minimal edits to `src/PyParticle/analysis.py` to keep backward compatibility: either import
   the new functions from `viz.data_prep` and re-export them, or leave thin wrappers that call
   the new module (avoid duplicating logic).
3) Unit tests added/updated under `tests/unit/` that validate the new data functions produce
   expected shapes and axis scales for typical inputs (happy path) plus 1-2 edge cases.
4) A short `reports/data_prep_migration.md` describing the changes, the new function signatures,
   and how to run the tests and demos to validate.

Context & constraints (read carefully)
--------------------------------------
- This repository uses src-layout: package root is `src/PyParticle`.
- Many viz helpers call a high-level `compute_variable(particle_population, varname, var_cfg, return_plotdat=True)`
  which returns (x, y, labs, xscale, yscale) when `return_plotdat=True`. That contract must be preserved
  for compatibility with plotting code, or the plotting code should be updated in the same patch.
- The new module should separate "data types" conceptually: distribution-like (dNdlnD), spectrum-like (Nccn vs s),
  and optical arrays (b_ext vs wvl or RH). Each function should return data in a small, consistent struct-like
  dict or tuple documented in the docstrings.
- Avoid adding heavy new dependencies. Use existing imports (numpy, scipy where already used) and keep changes minimal.
- Keep public API stability: tests and examples import from package root (e.g., `from PyParticle import analysis`) in some places — ensure import paths still work.

Key functions currently in `src/PyParticle/analysis.py` (source snapshot attached in the repo)
----------------------------------------------------------------------------------------
- compute_particle_variable
- compute_dNdlnD
- compute_Nccn
- compute_optical_coeffs
- compute_variable (high-level plotting accessor)
- build_default_var_cfg

Desired new organization (suggested)
-----------------------------------
- `src/PyParticle/viz/data_prep.py` exposing:
  - prepare_dNdlnD(particle_population, var_cfg) -> {"x": np.ndarray, "y": np.ndarray, "labs": [xlabel,ylabel], "xscale":"log"|"linear", "yscale":"linear"}
  - prepare_Nccn(particle_population, var_cfg) -> similar dict (x: s array, y: Nccn)
  - prepare_frac_ccn(...) -> dict
  - prepare_optical_vs_wvl(particle_population, var_cfg) -> dict (x: wvls, y: coeff)
  - prepare_optical_vs_rh(...)
  - prepare_Ntot(...) -> dict with y scalar or small array and labs

- Keep `compute_variable(...)` as a thin wrapper that dispatches to the matching `prepare_*` function.

API guidance and examples (required in the prompt)
-------------------------------------------------
For each `prepare_*` function, include in the docstring:
- Input types and minimal valid example `var_cfg` contents.
- Output format (dict keys: "x", "y", "labs", "xscale", "yscale"). Use None for x when plotting scalars.

Example: prepare_dNdlnD

Inputs:
- particle_population: a ParticlePopulation instance (has attributes `.ids`, `.num_concs`, and `.get_particle(part_id)`) used by existing code.
- var_cfg: dict with keys like `wetsize`, `normalize`, `method`, `N_bins`, `D_min`, `D_max`, `diam_scale`.

Output example:
{
  "x": np.ndarray(shape=(N_bins,)),  # bin centers
  "y": np.ndarray(shape=(N_bins,)),  # dNdlnD values
  "labs": ["D (m)", "dN/dlnD (1/m^3)"],
  "xscale": "log",
  "yscale": "linear",
}

Compatibility notes
-------------------
- Plotting code expects `compute_variable(..., return_plotdat=True)` to return a 5-tuple (x, y, labs, xscale, yscale). If you elect to change that contract, update `src/PyParticle/viz/plotting.py` and other callers in one patch so tests remain green.
- Avoid breaking public imports: `from PyParticle import analysis` should still work. Option: leave `src/PyParticle/analysis.py` with wrapper functions that import the new functions from `viz.data_prep` and return the same outputs.

Testing requirements
--------------------
- Add unit tests under `tests/unit/test_data_prep.py` covering:
  1. dNdlnD happy path: small synthetic ParticlePopulation (we can create a minimal fake population object in the test) and assert shapes and xscale == 'log'.
  2. Nccn happy path: small synthetic population and s_eval array; assert outputs have expected lengths and xscale == 'log'.
  3. Optical vs wvl: test that prepare_optical_vs_wvl returns x with given wvls and y of matching shape. Use monkeypatch to fake `build_optical_population` if needed.

Implementation notes for the other AI
------------------------------------
- Prefer small, reviewable commits that add tests first (failing), then implementation that makes them pass.
- Where helpful, include short inline comments explaining why a compatibility wrapper is left in `analysis.py`.
- Produce a single patch file or a small series of patches (git format) that can be applied directly.

What I (this AI) will do after you respond
-----------------------------------------
- Review the patch you produce. If it looks good, I'll apply it to the workspace, run the unit tests, and iterate on any failures.

How to deliver the patch to me
------------------------------
Provide either:
- A git-style patch/diff (unified diff) that I can apply with `git apply` or
- A set of modified files (file contents) and a short commit log that I can apply using the `apply_patch` tool.

Thank you — please produce a precise implementation plan and the patch (or files) required to execute it.
