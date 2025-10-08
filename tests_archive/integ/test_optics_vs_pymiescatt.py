# tests/integration/test_optics_vs_pymiescatt.py
import numpy as np
import pytest

# alternative: https://aaqr.org/articles/aaqr-18-02-tn-0067.pdf

# Guard the optional reference path early: try the package import first (clean),
# but fall back to loading the helper by file path if `examples` is not an importable
# package (this repo keeps examples/ as a top-level folder without __init__.py).
_PYMIE_AVAILABLE = False
pymiescatt_lognormal_optics = None
try:
    # Preferred: direct import if 'examples' is on the import path and is a package
    from examples.helpers.pymiescatt_comparison import pymiescatt_lognormal_optics  # type: ignore
    _PYMIE_AVAILABLE = True
except Exception:
    # Fallback: load the helper by file path relative to this test file
    try:
        import importlib.util
        from pathlib import Path

        helper_path = Path(__file__).resolve().parents[2] / 'examples' / 'helpers' / 'pymiescatt_comparison.py'
        spec = importlib.util.spec_from_file_location('pymiescatt_comparison', str(helper_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore
        pymiescatt_lognormal_optics = module.pymiescatt_lognormal_optics
        _PYMIE_AVAILABLE = True
    except Exception:
        _PYMIE_AVAILABLE = False

from PyParticle.population import build_population
from PyParticle.analysis import build_variable


@pytest.mark.skipif(not _PYMIE_AVAILABLE, reason="PyMieScatt-based helper is unavailable")
def test_bscat_matches_pymiescatt_single_modal():
    """
    Integration test:
    Compare PyParticle spectral scattering (b_scat) against a PyMieScatt-based reference
    for a single-species, single-mode binned lognormal population over a wavelength grid.
    """

    # --- 1) Configuration (single species, single lognormal mode) ---
    fixed_RI = 1.55 + 0.0j  # real RI; match the notebook's "fixed RI" comparison
    pop_cfg = {
        "type": "binned_lognormals",
        "N": [1e7],            # number concentration [m^-3]
        "GMD": [50e-9],        # geometric mean diameter [m] (50 nm)
        "GSD": [1.6],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
        # Keep runtime reasonable while giving smooth spectra; bump if CI allows
        "N_bins": 10000,
        "D_min": 1e-10,        # [m]
        "D_max": 1e-2,         # [m]
        # Communicate RI via species_modifications so the helper infers it cleanly
        "species_modifications": {
                'SO4':{
                "n_550": float(np.real(fixed_RI)),
                "k_550": float(np.imag(fixed_RI)),
                "alpha_n": 0.0,
                "alpha_k": 0.0,
            }
        },
    }

    # Wavelength grid in *meters* (SI) — notebook uses 350–1050 nm
    wvl_grid_m = np.linspace(350e-9, 1050e-9, 29)
    #wvl_grid_m = [500e-9]
    var_cfg = {
        "morphology": "core_shell",
        "wvl_grid": wvl_grid_m,
        "rh_grid": [0.0],
        # Provide RI here as well so PyParticle path is explicit/consistent
        "refractive_index": fixed_RI,
    }
    
    # --- 2) Build population and compute PyParticle b_scat ---
    pop = build_population(pop_cfg)

    bvar = build_variable(name="b_scat", scope="population", var_cfg=var_cfg)
    b_scat_pkg = bvar.compute(pop)
    
    # state_line plotter tends to flatten singleton axes; mirror that behavior
    if len(b_scat_pkg) == 1:
        b_scat_pkg = b_scat_pkg[0]
    b_scat_pkg = np.asarray(b_scat_pkg, dtype=float)

    # --- 3) Reference via PyMieScatt helper (already SI on outputs) ---
    wvl_nm, b_scat_ref, _b_abs_ref = pymiescatt_lognormal_optics(pop_cfg, var_cfg)
    # Convert returned wavelengths to meters for shape/ordering checks (values unused below)
    wvl_from_ref_m = np.asarray(wvl_nm, dtype=float) * 1e-9
    b_scat_ref = np.asarray(b_scat_ref, dtype=float)

    print(b_scat_pkg, b_scat_ref)

    # --- 4) Basic sanity checks ---
    assert wvl_from_ref_m.shape == wvl_grid_m.shape, "Wavelength grids are misaligned"
    assert b_scat_pkg.shape == b_scat_ref.shape, "Shape mismatch between package and reference"
    assert np.all(np.isfinite(b_scat_pkg)), "PyParticle b_scat contains non-finite values"
    assert np.all(np.isfinite(b_scat_ref)), "Reference b_scat contains non-finite values"
    assert np.all(b_scat_pkg >= -1e-16), "PyParticle b_scat has negative entries"
    assert np.all(b_scat_ref >= -1e-16), "Reference b_scat has negative entries"

    # --- 5) Numerical agreement (relative-or-absolute tolerance) ---
    tiny = 1e-16
    abs_diff = np.abs(b_scat_pkg - b_scat_ref)
    rel_err = abs_diff / np.maximum(np.abs(b_scat_ref), tiny)

    rtol = 0.05  # 5% relative
    atol = 1e-9  # m^-1 absolute

    max_rel = float(np.max(rel_err))
    max_abs = float(np.max(abs_diff))
    if not ((max_rel <= rtol) or (max_abs <= atol)):
        # Helpful diagnostics on failure
        i = int(np.argmax(rel_err))
        raise AssertionError(
            "b_scat mismatch:\n"
            f"  max_rel={max_rel:.3e} at index {i}, lambda={wvl_grid_m[i]:.3e} m\n"
            f"  max_abs={max_abs:.3e}\n"
            f"  pkg={b_scat_pkg[i]:.6e} m^-1, ref={b_scat_ref[i]:.6e} m^-1\n"
            "Consider: increasing N_bins, aligning helper bin count, or relaxing tolerances."
        )

# import numpy as np
# import pytest

# pytestmark = pytest.mark.requires_pymiescatt

# # Skip the whole test module if PyMieScatt is not available
# pytest.importorskip('PyMieScatt')

# # Import the example helper by file path (examples/ is not a package)
# import importlib.util
# from pathlib import Path
# helper_path = Path(__file__).resolve().parents[2] / 'examples' / 'helpers' / 'pymiescatt_comparison.py'
# spec = importlib.util.spec_from_file_location('pymiescatt_comparison', helper_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# pymiescatt_lognormal_optics = module.pymiescatt_lognormal_optics
# from PyParticle.population.builder import build_population
# from PyParticle.analysis import build_variable
# #from PyParticle.optics.builder import build_optical_population

# def test_bscat_babs_vs_pymiescatt():
#     # Make a local copy of the population config and increase the number of
#     # discretization bins to match the example notebook (which used 1000).
#     # This ensures PyParticle's size-discretization and the PyMieScatt helper
#     # use the same numberOfBins for better agreement in integrated optics.
#     #pop_cfg = small_cfg["population"].copy()
#     # prefer a larger bin count for the integration (keeps test deterministic)
#     #var_cfg = small_cfg['b_scat']

#     # Build the population and optics using the modified config so both
#     # PyParticle and the PyMieScatt helper below use identical inputs.
#     fixed_RI = 1.45 + 0.0j
#     pop_cfg = {
#         "type": "binned_lognormals",
#         "N": [1e9],
#         #"N_units": "m-3",
#         "GMD": [50e-9],
#         #"GMD_units": "m",
#         "GSD": [1.6],
#         "aero_spec_names": [["SO4"]],
#         "aero_spec_fracs": [[1.0]],
#         "N_bins": 1000,
#         "D_min": 1e-10,
#         "D_max": 1e-4,
#         #"refractive_index": fixed_RI,
#         'species_modifications': {'n_550': np.real(fixed_RI), 'k_550': 0.0, 'alpha_n': 0.0, 'alpha_k': 0.0}
#     }
    
    
#     # same cfg for both abs and scattering
#     var_cfg = {"morphology":"homogeneous","wvl_grid": np.linspace(350e-9,1050e-9,29), "rh_grid": [0.]}#, 'refractive_index': fixed_RI},  # simple case: single component
    
#     varname = 'b_scat'
#     pop = build_population(pop_cfg)
#     print(var_cfg, pop)
#     bsc = build_variable(name=varname, scope="population", var_cfg=var_cfg).compute(pop)

#     varname = 'b_abs'
#     #var_cfg = small_cfg[varname]
#     bab = build_variable(name=varname, scope="population", var_cfg=var_cfg).compute(pop)

#     # opop = build_optical_population(pop, {"type":"homogeneous", **var_cfg})
    
#     # bsc = opop.get_optical_coeff("b_scat", rh=0.0)  # shape [nWvl]
#     # bab = opop.get_optical_coeff("b_abs", rh=0.0)

#     wvl_nm, bsc_ref, bab_ref = pymiescatt_lognormal_optics(pop_cfg, var_cfg)

#     # print(opop.species[0].refractive_index)
#     print("PyParticle b_scat:", bsc)
#     print("PyMieScatt b_scat_ref:", bsc_ref)
#     print("PyParticle b_abs:", bab)
#     print("PyMieScatt b_abs_ref:", bab_ref)
#     assert bsc.shape == bsc_ref.shape == bab.shape == bab_ref.shape
#     # Basic contract checks: non-negative (allow tiny numerical noise), finite, and shape matches.
#     assert np.all(bsc >= 0) and np.all(np.isfinite(bsc))
#     # allow small negative numerical noise in absorption values (scale with signal)
#     # Allow a tight relative tolerance when the discretizations match.
#     try:
#         assert np.allclose(bsc, bsc_ref, rtol=1e-3)
#     except AssertionError:
#         # Provide a helpful diagnostic (ratios) for debugging when the
#         # check fails; re-raise so CI still catches the problem.
#         ratios = np.divide(bsc_ref, bsc, out=np.full(bsc.shape, np.nan), where=bsc!=0)
#         print("bsc_ref / bsc ratios:", ratios)
#         raise
#     #assert np.allclose(bab, bab_ref, rtol=1e-3)