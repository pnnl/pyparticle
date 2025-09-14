import numpy as np

import PyParticle
from PyParticle.population import build_population
from PyParticle.optics.builder import build_optical_population


def main():
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
        "D_min": 0.02,                           # micrometers
        "D_max": 1.00,                           # micrometers
        "N_bins": 40,
        "N": [1500.0],                           # total number concentration per mode
        "GMD": [0.10],                           # micrometers
        "GSD": [1.6],
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

    # -------------------------------
    # 2) Build homogeneous optics
    # -------------------------------
    rh_grid = np.array([0.0, 0.5, 0.8, 0.9, 0.95])
    wvl_grid_um = np.array([0.47, 0.55, 0.66])   # micrometers
    wvl_grid_m = wvl_grid_um * 1e-6              # convert to meters

    # Notes:
    # - type must be "homogeneous"
    # - diameter_units is "um" to match the base population diameter getters
    # - n_550/alpha_n, k_550/alpha_k parameterize a simple RI spectrum for the homogeneous sphere
    optics_config = {
        "type": "homogeneous",
        "rh_grid": rh_grid,
        "wvl_grid": wvl_grid_m,
        "compute_optics": True,
        "temp": 293.15,
        "diameter_units": "um",
        # Optional: simple homogeneous RI spectrum (override defaults if desired)
        # Here we pick a weakly absorbing homogeneous sphere
        "n_550": 1.54,
        "alpha_n": 0.0,
        "k_550": 0.00,
        "alpha_k": 0.0,
        # Fallback behavior if PyMieScatt isn't installed
        "single_scatter_albedo": 0.9,
        "fallback_kappa": 0.6,  # use sulfate-like growth in fallback path
        # Pass through species_modifications if your homogeneous RI builder wants it later
        "species_modifications": binned_lognormals_config["species_modifications"],
        # "specdata_path": "/path/to/your/species_data/",  # optional
    }

    opt_pop = build_optical_population(pop, optics_config)

    # -------------------------------
    # 3) Example: retrieve optical coefficients
    # -------------------------------
    bext = opt_pop.get_optical_coeff("ext")  # shape (len(RH), len(λ))
    print("b_ext grid shape:", bext.shape)
    print("b_ext grid:\n", bext)

    print("\nPointwise b_ext(rh, λ):")
    for rh in rh_grid:
        for wvl in wvl_grid_m:
            val = opt_pop.get_optical_coeff("ext", rh=rh, wvl=wvl)
            print(f"  rh={rh:4.2f}, λ={wvl*1e6:5.2f} µm: b_ext = {val:.6g}")

    # Also show slices
    rh_sel = 0.5
    wvl_sel = 0.55e-6
    print("\nb_ext at RH=0.5 across wavelengths:", opt_pop.get_optical_coeff("ext", rh=rh_sel))
    print("b_ext at λ=0.55 µm across RH:      ", opt_pop.get_optical_coeff("ext", wvl=wvl_sel))

    # Asymmetry parameter example
    g_grid = opt_pop.get_optical_coeff("g")
    print("\nAsymmetry parameter g grid:\n", g_grid)


if __name__ == "__main__":
    main()