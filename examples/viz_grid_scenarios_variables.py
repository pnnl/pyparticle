"""Example demonstrating make_grid_scenarios_variables_same_timestep.

This script is runnable and quiet except for writing out the figure when run as
__main__.
"""
from pathlib import Path

from PyParticle.viz import make_grid_scenarios_variables_same_timestep


def main():
    # use one on-disk config (binned_lognormals) and one inline dict (monodisperse)
    scenarios = [
        "examples/configs/binned_lognormal.json",
        {"type": "monodisperse", "aero_spec_names": ["SO4"], "aero_spec_fracs": [[1.0]], "N": [1000.0], "D": [0.1]},
    ]
    variables = ["dNdlnD", "frac_ccn"]
    timestep = 3600

    # load scenario files and extract 'population' sub-dicts when present
    resolved_scenarios = []
    import json, yaml
    from pathlib import Path
    for s in scenarios:
        if isinstance(s, dict):
            resolved_scenarios.append(s)
            continue
        p = Path(s)
        if p.exists():
            txt = p.read_text()
            try:
                doc = json.loads(txt)
            except Exception:
                doc = yaml.safe_load(txt)
            if isinstance(doc, dict) and "population" in doc:
                resolved_scenarios.append(doc["population"])
            else:
                resolved_scenarios.append(doc)
        else:
            resolved_scenarios.append(s)

    fig, axarr = make_grid_scenarios_variables_same_timestep(
        resolved_scenarios,
        variables,
        timestep,
        var_cfg_overrides={
            "frac_ccn": {"s_eval": [0.05, 0.1, 0.2]},
            "b_ext": {"wvl_select": 0.55e-6, "rh_select": 0.5},
        },
        figsize=(12, 9),
        hspace=0.35,
        wspace=0.25,
    )

    out = Path(__file__).parent / "out_grid_scenarios_variables.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
