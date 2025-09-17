"""Demo: build a PartMC population and plot a variable using PyParticle.viz

This script is intended to be run from the repository root with the
`pyparticle` conda environment active:

  conda activate pyparticle
  python examples/partmc_demo.py

It will read `examples/configs/partmc_example.json`, build the population
via `build_population`, and use `PyParticle.viz.plot_lines` to plot a single
variable (the default size distribution variable) and save `examples/partmc_demo.png`.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

from PyParticle.population import build_population
from PyParticle.viz import make_grid, plot_lines, format_axes, add_legend


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "examples" / "configs" / "partmc_example.json"
    out_png = repo_root / "examples" / "partmc_demo.png"
    cfg = json.load(open(cfg_path))

    # Build three populations from the same scenario directory at different timesteps
    times = [1, 12, 24]
    pops = []
    for t in times:
        cfg_t = dict(cfg)
        cfg_t['timestep'] = t
        pops.append(build_population(cfg_t))

    # Create a 1x3 grid
    fig, axarr = make_grid(1, 3, figsize=(12, 4))

    for i, (t, pop) in enumerate(zip(times, pops)):
        ax = axarr[0, i]
        line, labs = plot_lines("dNdlnD", (pop,), var_cfg=None, ax=ax)
        title = f"t={t}"
        # compute_variable labels are returned as a list; try to extract x/y labels
        xlabel = labs[0] if isinstance(labs, (list, tuple)) and len(labs) > 0 else ""
        ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else ""
        format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_xscale("log")
        add_legend(ax)

    fig.savefig(out_png)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()
