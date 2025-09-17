ONBOARDING — quick start
========================

Quick pointer for the next AI or developer who opens this repository.

Primary handoff file: `AI_NEXT_STEPS.md` (branch: `feat/viz-grids`)

One-line onboarding command (run from repo root, zsh):

```bash
conda env create -f environment-dev.yml -n pyparticle-dev && conda activate pyparticle-dev && python -c "import PyParticle; print('import OK')"
```

Notes
-----
- After the environment is active, open `AI_NEXT_STEPS.md` and follow the "Top priorities" section.
- To run the PartMC demo (if PartMC outputs are available):

```bash
python examples/viz_grid_partmc_scenarios.py
```

If you want a PR landing page, open a draft PR for branch `feat/viz-grids` on GitHub and point reviewers to `AI_NEXT_STEPS.md` and `ONBOARDING.md`.
