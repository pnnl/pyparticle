from pathlib import Path
from PyParticle.viz import make_grid_scenarios_variables_same_timestep

def main():
    repo_root = Path(__file__).resolve().parent
    scenarios = [str(repo_root / "configs" / "binned_lognormal.json"),
                 {"type": "monodisperse", "aero_spec_names": ["SO4"], "aero_spec_fracs": [[1.0]], "N": [1000.0], "D": [0.1]}]
    variables = ["dNdlnD", "frac_ccn"]
    fig, axarr = make_grid_scenarios_variables_same_timestep(scenarios, variables, timestep=0)
    for i in range(axarr.shape[0]):
        for j in range(axarr.shape[1]):
            ax = axarr[i, j]
            print(f"ax[{i},{j}] xscale={ax.get_xscale()} yscale={ax.get_yscale()}")

if __name__ == '__main__':
    main()
