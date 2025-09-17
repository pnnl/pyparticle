
AI next steps — concise, prioritized handoff (branch: feat/viz-grids)
=================================================================

Purpose
-------
Clear, actionable instructions for the next AI. Priorities are first; lower-priority items are "Do later" and require explicit approval before starting.

Top priorities (act on now)
--------------------------
1) Make grid examples runnable and correct
   - Update PartMC example config(s) to accept an ensemble directory (parent containing runs) and an optional `member` field.
   - Update example script(s) to detect the ensemble, select a run, and print the chosen path.
   - Verify by running the PartMC demo and confirm a PNG is written.

2) Fix x/y axis scale handling for line plots
   - Ensure plotting order: plot -> set_xscale/set_yscale -> format_axes -> add_legend.
   - Add a unit test that asserts `ax.get_xscale()` == 'log' for a dNdlnD plot.
   - Add a demo verification that prints each subplot's x/y scale.

Rules for priority work
----------------------
- Work on these items first. Create a focused branch (e.g. `feat/viz-grids-priority`) for changes.
- Make minimal, test-driven edits: add or update one unit test, implement the fix, run that test and the demo.
- Commit small, atomic changes and push the branch. Record verification steps in `reports/onboarding_summary.md`.

Do later (ask before starting)
----------------------------
A) Second-line plotting (plot two lines on same axis; UX improvements)
B) Scatter plot helper (or extend plotting with `plot_kind='scatter'`)
C) AMBRS integration (adapter/submodule) — requires approval for new deps
D) Cleanup temporary/debug scripts and files — delete only after examples are stable
E) CI enhancements to run PartMC/AMBRS tests (gate by env vars)

Do-later protocol
-----------------
- Stop and ask: "May I proceed to Do-later task <A..E>?" and wait for explicit approval.
- If human approval is not available, open a PR with the proposed change and label `do-later/waiting-for-approval`.

Minimal verification commands
-----------------------------
Run these from the repo root after activating the dev env:

```bash
# quick import check
python -c "import PyParticle; print('import OK')"

# run single unit test (xscale)
pytest -q tests/unit/test_grid_scenarios_variables.py::test_xscale_log

# run partmc demo (writes PNG)
python examples/viz_grid_partmc_scenarios.py
```

If you want me to start the priority work now, say "Do priorities now" and I'll create the branch, implement the edits, run the tests and demos, and push a verification branch with results.


How to use
----------
- Check out the branch: git checkout feat/viz-grids
- Recreate the conda environment used for testing (recommended):
  - conda env create -f environment-dev.yml
  - conda activate pypparticle
- Run the quick smoke demos (examples):
  - python examples/viz_grid_scenarios_variables.py
  - python examples/viz_grid_partmc_scenarios.py  # requires PartMC outputs

Quick checklist (high priority)
-------------------------------
1. Remove remaining noisy prints / warnings from library code.
   - Files touched: src/PyParticle/viz/plotting.py (already cleaned),
     src/PyParticle/aerosol_particle.py (surface-tension warning comes from model code and
     may be okay to keep), population factories (some prints removed).

2. Add unit tests for new helper(s).
   - tests/unit/test_grid_scenarios_variables.py exists but expand to cover:
     - optics variables (requires optics builder available in env)
     - scalar variables (Ntot), and edge cases (empty populations)
   - Add integration test(s) gated by env var PYPARTICLE_PARTMC_RUNS.

3. Align example configs & species names.
   - Some example configs reference species names (e.g., NaCl) that do not match the bundled datasets.
   - Options:
     - Update example configs to use canonical species names in datasets/.
     - Or add a mapping step in example/demonstration scripts (temporary hack).

4. Optics: make sure optics morphology types used in examples exist in the environment
   (core-shell may be missing in some environments). If optics builder raises errors,
   update examples to avoid optics variables or ensure optics builder is installed.

5. CI: add a minimal CI job that runs the new unit tests in a matrix with and without
   network (PartMC runs) and sets PYPARTICLE_DATA_PATH appropriately.

Implementation details & tips
-----------------------------
- Helper API to look at:
  - `make_grid_scenarios_variables_same_timestep(...)` in `src/PyParticle/viz/grid_scenarios_variables.py`
  - `plot_lines(var, (pop,), var_cfg, ax=...)` — it expects `compute_variable` to return plotting arrays.
  - `build_population(cfg)` builds populations from dicts; scenario JSON/YAML files sometimes wrap the
    actual population under a `population` key.

- When building populations for a timestep, set `cfg['timestep'] = <value>` *before* calling `build_population`.

- Common runtime issues observed during development:
  - TypeError from matplotlib when invalid kwargs are passed to fig.subplots_adjust — fixed in `viz.layout.make_grid`.
  - Missing PartMC `netCDF4` package will cause the 'partmc' factory to raise ModuleNotFoundError; CI should account for that.
  - Optics builder errors for unknown morphology (e.g., 'core-shell') — ensure optics dependencies are installed.

Files & areas to inspect (fast map)
----------------------------------
- New helper: src/PyParticle/viz/grid_scenarios_variables.py
- Grid helpers: src/PyParticle/viz/grids.py (existing helpers)
- Plotting plumbing: src/PyParticle/viz/plotting.py
- Layout utilities: src/PyParticle/viz/layout.py
- Analysis helpers: src/PyParticle/analysis.py (compute_variable, build_default_var_cfg)
- Examples: examples/viz_grid_scenarios_variables.py, examples/viz_grid_partmc_scenarios.py

Commands the next agent should run locally
----------------------------------------
1. Quick lint / static checks
   - python -m pip install -r requirements-test.txt
   - python -m pyflakes src/PyParticle || flake8 src/PyParticle

2. Run unit tests (fast)
   - pytest -q tests/unit/test_grid_scenarios_variables.py

3. Run demos (visual smoke)
   - python examples/viz_grid_scenarios_variables.py
   - python examples/viz_grid_partmc_scenarios.py  # only if PartMC runs are available

PR description template (copy into PR web UI)
--------------------------------------------
Summary: Add grid helper `make_grid_scenarios_variables_same_timestep`, plus examples and tests. This is a WIP
PR that adds the helper and example workflows; follow-ups include adding more unit tests, aligning example species,
and polishing the plotting output.

Known issues / TODO
-------------------
- Tidy remaining example configs to use canonical species names.
- Add optics-enabled CI run or guard optics tests when builder is absent.
- Expand unit test coverage for scalar and optics variables.

PRIORITY vs DO LATER (workflow policy for the next AI)
-----------------------------------------------------
Summary: The items in this file have been reorganized so a future AI can immediately start on the highest-impact tasks, and defer lower-priority work until requested.

Priority tasks (start immediately)
- Goal: Make the grid examples runnable and visually correct. These must be completed before any other edits.

1) Fix `partmc_example.json` and related grid example configs
   - Make the `partmc` example accept an ensemble directory (parent containing multiple runs) and optionally a `member` field.
   - Update example scripts to detect ensemble directories and choose a member when needed. Print the selected run path for debugging.
   - Validate by running `python examples/viz_grid_partmc_scenarios.py` (or the specific demo) and reporting which run was chosen.
   - If changing configs, create a small unit test that verifies the config is parsed into a canonical dict passed to `build_population`.

2) Fix x/y axis scale handling for line plots
   - Ensure `plot_lines`/`viz.plotting` applies axis scales after plotting (plot -> set_xscale/set_yscale -> format_axes) and does not have its output unintentionally overridden by formatting utilities.
   - Add a unit test asserting `ax.get_xscale()` equals `'log'` for a dNdlnD plot and that both lines appear when two line calls are made.
   - Quick verification script: call the helper that builds the grid for scenarios × variables at a chosen timestep and print axis scales for each subplot.

Work protocol for priorities
- Create a branch named `feat/viz-grids-priority-<ticket>` for these edits if you're not already on one.
- Make very small, test-driven changes: add or update one unit test, implement the minimal fix, run that test and the demo example locally.
- Commit with a focused message and push the branch.
- After both priority items are green (tests and local demo runs), produce a short verification note in `reports/onboarding_summary.md` describing what changed and how you validated it.

Do later (ask before starting)
- These items should only be tackled after the priority tasks are completed and only with explicit approval from a human or an integrating AI.

A) Add second-line plotting UX improvements
   - Provide examples and optionally extend `plot_lines` to accept multiple var_cfgs at once.

B) Scatter plot helper
   - Implement `plot_scatter` or extend `plot_lines` with `plot_kind='scatter'` and add a demo script mirroring the line-grid example.

C) AMBRS integration
   - Add adapter or instructions to consume AMBRS outputs. This may require a submodule or new dependency; get approval before adding.

D) Cleanup unneeded debug code and retired examples
   - Remove temporary scripts only after the new examples are stable and their PNG outputs are checked into the repo (if desired).

E) CI enhancements
   - Add CI jobs for the new tests and ensure PartMC/AMBRS-dependent tests are gated by env vars or separate CI matrices.

Do-later workflow rules
- Before starting any Do-later task, stop and ask: "May I proceed to Do-later task <letter> (A..E)?"
- Wait for explicit approval. If a human isn't available, create a short PR with the proposed change and label it 'do-later/waiting-for-approval'.

Notes for the future AI
- Prioritize reproducibility: always run tests and demo scripts in a fresh environment or a reproducible conda env.
- Keep commits small and atomic; include test additions alongside fixes.
- If a change may break examples for users without PartMC runs, guard it with environment-detection or a toggled flag.

If you'd like, I can now implement the top-priority changes (1 and 2): edit `examples/configs/partmc_example.json` to use an ensemble dir + `member` field, add the unit test for x/y scaling, and modify `src/PyParticle/viz/plotting.py` if needed. Say "Do priorities now" and I'll start (I will create a branch, make the edits, run tests, and push the branch with results).
