import numpy as np
import json
from pathlib import Path
import PyParticle
from PyParticle.population import build_population
from PyParticle.optics.builder import build_optical_population

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Load configs
    config_dir = Path(__file__).parent / "configs"
    pop_cfg = read_json(config_dir / "binned_lognormal.json")
    optics_cfg = read_json(config_dir / "optics_homogeneous.json")
    
    # pop_cfg = read_json(Path("examples/configs/binned_lognormal.json"))
    # optics_cfg = read_json(Path("examples/configs/optics_homogeneous.json"))

    # Build population
    pop = build_population(pop_cfg)

    # Prepare optics config
    optics_cfg = dict(optics_cfg)  # make mutable copy
    # Convert wavelength grid to meters
    wvl_grid_um = np.array(optics_cfg.pop("wvl_grid_um"))
    optics_cfg["wvl_grid"] = wvl_grid_um * 1e-6
    # Add species_modifications from population config
    optics_cfg["species_modifications"] = pop_cfg.get("species_modifications", {})

    # Build optics
    opt_pop = build_optical_population(pop, optics_cfg)

    # Retrieve and print optical coefficients
    bext = opt_pop.get_optical_coeff("ext")
    print("b_ext grid shape:", bext.shape)
    print("b_ext grid:\n", bext)

    rh_grid = np.array(optics_cfg["rh_grid"])
    wvl_grid_m = optics_cfg["wvl_grid"]

    print("\nPointwise b_ext(rh, λ):")
    for rh in rh_grid:
        for wvl in wvl_grid_m:
            val = opt_pop.get_optical_coeff("ext", rh=rh, wvl=wvl)
            print(f"  rh={rh:4.2f}, λ={wvl*1e6:5.2f} µm: b_ext = {val:.6g}")

    # Slices
    rh_sel = 0.5
    wvl_sel = 0.55e-6
    print("\nb_ext at RH=0.5 across wavelengths:", opt_pop.get_optical_coeff("ext", rh=rh_sel))
    print("b_ext at λ=0.55 µm across RH:      ", opt_pop.get_optical_coeff("ext", wvl=wvl_sel))

    g_grid = opt_pop.get_optical_coeff("g")
    print("\nAsymmetry parameter g grid:\n", g_grid)

if __name__ == "__main__":
    main()