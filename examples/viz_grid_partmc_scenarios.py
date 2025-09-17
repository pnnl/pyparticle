"""Example: grid of PartMC-based scenarios (rows) × variables (cols).

This example follows the pattern in `examples/partmc_demo.py` but builds
three PartMC-based populations (e.g., runs '01', '05', '10') and plots
variables across a single common timestep using
`make_grid_scenarios_variables_same_timestep`.

Notes:
- This script expects PartMC output directories to be available at
  `examples/configs/partmc_runs/01`, `.../05`, `.../10`. If they are not
  present, the script raises a clear error explaining how to provide them.
"""
from pathlib import Path
import json

from PyParticle.population import build_population
from PyParticle.viz import data_prep
from PyParticle.viz.data_prep import build_default_var_cfg
import matplotlib.pyplot as plt
import numpy as np


def main():
    repo_root = Path(__file__).resolve().parent.parent
    base_cfg_path = repo_root / "examples" / "configs" / "partmc_example.json"
    base_cfg = json.load(open(base_cfg_path))

    # Desired partmc run folders (these should contain PartMC output files)
    run_ids = ["01", "05", "10"]
    runs_dir = Path("/Users/fier887/Downloads/partmc_runs/")
    scenarios = []
    for rid in run_ids:
        run_dir = runs_dir / rid
        if not run_dir.exists():
            raise ValueError(
                f"PartMC run directory not found: {run_dir}\n"
                "Provide PartMC outputs at examples/configs/partmc_runs/<id>/ or edit the example to point to your data."
            )
        cfg = dict(base_cfg)
        cfg["partmc_dir"] = str(run_dir)
        # keep other fields from base_cfg (repeat, species_modifications, etc.)
        scenarios.append(cfg)

    variables = ["dNdlnD", "frac_ccn"]
    timestep = base_cfg.get("timestep", 0)

    # Build populations and precompute plot-ready data using data_prep.
    pops = []
    for cfg in scenarios:
        cfg_local = dict(cfg)
        cfg_local["timestep"] = timestep
        pop = build_population(cfg_local)
        pops.append((cfg_local, pop))

    nrows = len(pops)
    ncols = len(variables)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    for i, (cfg_local, pop) in enumerate(pops):
        for j, var in enumerate(variables):
            ax = axes[i, j]
            # merge defaults
            var_cfg = build_default_var_cfg(var)
            # allow variable-specific overrides if desired (not used here)
            if var == "dNdlnD":
                dat = data_prep.prepare_dNdlnD(pop, var_cfg)
            elif var == "frac_ccn":
                dat = data_prep.prepare_frac_ccn(pop, var_cfg)
            else:
                dat = data_prep.prepare_Nccn(pop, var_cfg)

            x, y, labs = dat["x"], dat["y"], dat["labs"]
            if x is None:
                ax.bar([0], y)
            else:
                ax.plot(x, y)
            ax.set_xscale(dat.get("xscale", "linear"))
            ax.set_yscale(dat.get("yscale", "linear"))
            ax.set_xlabel(labs[0])
            if j == 0:
                ax.set_ylabel(labs[1])

    out = repo_root / "examples" / "out_grid_partmc_scenarios.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
