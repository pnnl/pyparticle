# Visualization (viz)

← Back to Index

Overview

The `PyParticle.viz` subpackage provides helpers for plotting populations and optical results. High-level grid helpers build populations from config dicts when needed, call plotting primitives, and return `(fig, ax)` so callers can save or further format the figure.

Simple line plot example

```python
from PyParticle.viz.builder import build_plotter
cfg = {"varname": "b_scat", "var_cfg": {"wvl_grid": [550e-9], "rh_grid": [0.0]}}
plotter = build_plotter("state_line", cfg)
fig, ax = plotter.plot(population, label="example")
fig.savefig("out_bscat.png")
```

Design notes

- Plotting helpers do not set axis labels by default. Formatting utilities live in `viz/formatting.py` and are responsible for labels/units. This separation allows composable figure generation where titles, legends, and labels are applied in a single place.
- Grid helpers accept either pre-built `ParticlePopulation` objects or configuration dictionaries; when given config dicts they will call `build_population` internally before plotting.

Saving figures

Use `fig.savefig("filename.png", dpi=150)` to save figures in non-interactive environments (CI or headless servers). The test suite forces a headless Matplotlib backend during pytest runs.
