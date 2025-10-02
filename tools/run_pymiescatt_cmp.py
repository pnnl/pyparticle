"""Runner for PyMieScatt comparison helper

This script calls examples.helpers.pymiescatt_comparison.pymiescatt_lognormal_optics
with a small, single-species lognormal population and prints results.
"""
import traceback
import numpy as np

try:
    from examples.helpers.pymiescatt_comparison import pymiescatt_lognormal_optics
except Exception as e:
    print("Failed to import pymiescatt_comparison:")
    traceback.print_exc()
    raise SystemExit(1)

# Build a minimal single-mode population config (SI units where appropriate)
pop_cfg = {
    'GMD': [200e-9],        # geometric mean diameter: 200 nm (meters)
    'GSD': [1.6],           # geometric std dev
    'N': [1e9],             # number conc: 1e9 m^-3 = 1000 cm^-3
    'GMD_units': 'm',
    'N_units': 'm-3',
    'species_modifications': {
        'SO4': {'n_550': 1.52, 'k_550': 0.0}
    },
    'N_bins': 200,
    'D_min': 1e-9,
    'D_max': 2e-6,
}

# Variable config: wavelengths in meters
var_cfg = {
    'wvl_grid': np.array([550e-9, 650e-9], dtype=float)
}

print("Running PyMieScatt lognormal optics comparison with:")
print(pop_cfg)
print("wavelengths (m):", var_cfg['wvl_grid'])

try:
    wl_nm, b_scat_m, b_abs_m = pymiescatt_lognormal_optics(pop_cfg, var_cfg)
    np.set_printoptions(precision=6, suppress=True)
    print('\nResult:')
    print('wavelengths (nm):', wl_nm)
    print('b_scat (m^-1):', b_scat_m)
    print('b_abs  (m^-1):', b_abs_m)
except Exception:
    print("Error while running pymiescatt_lognormal_optics:")
    traceback.print_exc()
    raise SystemExit(2)
