# viz — Flexible Plotting Package

This small package provides a modular framework for creating figures in
Python using `matplotlib`. The goal is to separate layout, plotting,
styling and formatting concerns so they can be reused and extended.

Package contents

- `layout.py`: helpers to create figure layouts (subplots / GridSpec).
- `plotting.py`: functions that draw on a single `Axes` (e.g. `plot_line`).
- `styling.py`: color/linestyle helpers that return style values.
- `formatting.py`: functions to set titles/labels/limits/legends on axes.
- `demo.py`: a simple end-to-end example that produces `viz_demo.png`.

Guiding principles

- Separation of concerns: layout, plotting, styling, formatting live in
  distinct modules.
- Reusability: functions are small and composable.
- Flexibility: easy to add new plot types, styles, or layouts.

Quick example

```python
from PyParticle.viz import make_grid, plot_line, get_colors, format_axes, add_legend

fig, axes = make_grid(1, 2)
colors = get_colors(2)
ax = axes[0, 0]
plot_line(ax, x=[1,2,3], y=[2,4,1], color=colors[0], label="example")
format_axes(ax, xlabel="X", ylabel="Y", title="Demo")
add_legend(ax)
fig.savefig("demo.png")
```

Notes

- For now this package is intentionally self-contained under
  `src/PyParticle/viz` so you can adjust imports later when integrating
  with the wider project.
