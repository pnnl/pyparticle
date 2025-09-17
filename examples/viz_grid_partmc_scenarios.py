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
from PyParticle.viz import make_grid_scenarios_variables_same_timestep


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

    fig, axarr = make_grid_scenarios_variables_same_timestep(
        scenarios,
        variables,
        timestep,
        figsize=(10, 6),
        hspace=0.35,
        wspace=0.25,
    )

    out = repo_root / "examples" / "out_grid_partmc_scenarios.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
