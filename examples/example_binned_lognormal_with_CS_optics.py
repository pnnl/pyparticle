import numpy as np
import PyParticle
from PyParticle.population import build_population
from PyParticle.optics.builder import build_optical_population

# -------------------------------
# 1) Build the base population
# -------------------------------
# Single-mode, binned lognormal population
# - 'type' must be 'binned_lognormals'
# - For single-mode, use lists of length 1 for N, GMD, GSD; D_min/D_max can be scalars.
# - aero_spec_names MUST be a list-of-lists (single mode -> one inner list).
binned_lognormals_config = {
    "type": "binned_lognormals",
    "aero_spec_names": [["SO4", "BC"]],      # single-mode names as list-of-lists
    "aero_spec_fracs": [[0.85, 0.15]],       # per-mode fractions (list-of-lists)
    "D_min": 0.02,                           # consistent units with GMD/D_max (e.g., micrometers if you like)
    "D_max": 1.00,
    "N_bins": 40,
    "N": [1500.0],                           # total number concentration per mode
    "GMD": [0.10],                           # geometric mean diameter per mode
    "GSD": [1.6],                            # geometric std. dev. per mode
    "D_is_wet": False,                       # set True if diameters are wet sizes
    "species_modifications": {
        "BC": {
            "k_550": 0.8,      # imag(n) at 550 nm
            "alpha_k": 0.0,    # spectral slope for k
            "n_550": 1.85,     # real(n) at 550 nm
            "alpha_n": 0.0,    # spectral slope for n
            "density": 1800,   # kg/m^3
        },
        "SO4": {
            "k_550": 0.0,
            "alpha_k": 0.0,
            "n_550": 1.45,
            "alpha_n": 0.0,
            "density": 1770,   # kg/m^3
            "kappa": 0.6,      # hygroscopicity
        },
    },
}

pop = build_population(binned_lognormals_config)

import numpy as np
from PyParticle.population import build_population
from PyParticle.optics.builder import build_optical_population

# ... build `pop` with binned_lognormals_config ...

rh_grid = np.array([0.0, 0.5, 0.8, 0.9, 0.95])
wvl_grid_um = np.array([0.47, 0.55, 0.66])
wvl_grid_m = wvl_grid_um * 1e-6

optics_config = {
    "type": "core_shell",
    "rh_grid": rh_grid,
    "wvl_grid": wvl_grid_m,
    "compute_optics": True,
    "temp": 293.15,
    "specdata_path":'/Users/fier887/Library/CloudStorage/OneDrive-PNNL/Code/PyParticle/datasets/species_data/',
    "species_modifications": binned_lognormals_config["species_modifications"],
}

opt_pop = build_optical_population(pop, optics_config)

# Examples:
bext = opt_pop.get_optical_coeff("ext")                     # shape (len(RH), len(λ))

for rr,rh in enumerate(rh_grid):
    for ww,wvl in enumerate(wvl_grid_m):
        print(opt_pop.get_optical_coeff("ext", rh=rh, wvl=wvl),bext[rr,ww])
# bext_055 = opt_pop.get_optical_coeff("ext", rh=0.5, wvl=0.55e-6)  # scalar
# bext_055
