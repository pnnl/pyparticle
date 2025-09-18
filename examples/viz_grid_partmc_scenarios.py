"""Example: PartMC scenarios (rows) × variables (columns) using viz grid helpers.

Refactored to use `PyParticle.viz.grids.make_grid_popvars` so the example is
declarative: each row is a PartMC config dict; each column a variable name.

Requirements:
- PartMC output directories present under the specified `runs_dir` (edit path
  below to point to your local PartMC outputs). Each run folder should contain
  an `out/` directory with NetCDF files consistent with `partmc_example.json`.

This script intentionally avoids manual matplotlib subplot handling—axis
formatting, legends, and log-scaling heuristics come from the viz helpers.
"""

from pathlib import Path
import json
from PyParticle.viz.grids import make_grid_popvars
from PyParticle.analysis import compute_variable  # optional direct usage demo
import matplotlib.pyplot as plt


def build_partmc_scenarios(base_cfg_path, run_ids, runs_dir):
    """Return a list of PartMC config dicts (one per run id)."""
    base_cfg = json.load(open(base_cfg_path))
    scenarios = []
    for rid in run_ids:
        run_dir = Path(runs_dir) / rid
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Missing PartMC run directory: {run_dir}\n"
                "Provide outputs at <runs_dir>/<id>/ (with 'out/' inside) or update the path."
            )
        cfg = dict(base_cfg)
        cfg["partmc_dir"] = str(run_dir)
        scenarios.append(cfg)
    return scenarios, base_cfg.get("timestep", 0)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    base_cfg_path = repo_root / "examples" / "configs" / "partmc_example.json"

    # EDIT THIS to point to your PartMC run directories root
    runs_dir = Path("/Users/fier887/Downloads/partmc_runs/")
    run_ids = ["01", "05", "10"]

    scenarios, timestep = build_partmc_scenarios(base_cfg_path, run_ids, runs_dir)

    # Attach timestep to each scenario (each is a config dict consumed by builder)
    scenarios_with_time = []
    for cfg in scenarios:
        cfg_time = dict(cfg)
        cfg_time["timestep"] = timestep
        scenarios_with_time.append(cfg_time)

    variables = ["dNdlnD", "frac_ccn", 'b_scat']

    fig, axes = make_grid_popvars(
        scenarios_with_time,
        variables,
        var_cfg=None,  # could pass per-variable config mapping
        time=None,
        figsize=(4 * len(variables), 3 * len(scenarios_with_time)),
        sharex_columns=True,
        sharey_rows=True,
    )

    fig.suptitle("PartMC scenario comparison (timestep={})".format(timestep))
    fig.tight_layout()

    out = repo_root / "examples" / "out_grid_partmc_scenarios.png"
    fig.savefig(out, dpi=200)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
