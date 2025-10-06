import numpy as np
import pytest

# Try to import the updated helper; skip test if unavailable (e.g., PyMieScatt not installed)
try:
    from examples.helpers.pymiescatt_comparison import reference_optics_for_population
    _REF_AVAILABLE = True
except Exception:
    try:
        import importlib.util
        from pathlib import Path
        helper_path = Path(__file__).resolve().parents[1] / 'examples' / 'helpers' / 'pymiescatt_comparison.py'
        spec = importlib.util.spec_from_file_location('pymiescatt_comparison', str(helper_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference_optics_for_population = getattr(module, "reference_optics_for_population", None)
        _REF_AVAILABLE = reference_optics_for_population is not None
    except Exception:
        _REF_AVAILABLE = False

from PyParticle.population import build_population
from PyParticle.analysis import build_variable


@pytest.mark.skipif(not _REF_AVAILABLE, reason="PyMieScatt-based reference helper is unavailable")
def test_bscat_matches_reference_single_modal():
    fixed_RI = 1.55 + 0.0j
    pop_cfg = {
        "type": "binned_lognormals",
        "N": [1e7],            # m^-3
        "GMD": [50e-9],        # m
        "GSD": [1.6],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
        "N_bins": 400,         # ~fast for CI; increase for tighter match
        "D_min": 1e-10,
        "D_max": 1e-2,
        "species_modifications": {
            "SO4": {
                "n_550": float(np.real(fixed_RI)),
                "k_550": float(np.imag(fixed_RI)),
                "alpha_n": 0.0,
                "alpha_k": 0.0,
            }
        },
    }
    wvl_grid_m = np.linspace(350e-9, 1050e-9, 29)
    var_cfg = {
        "morphology": "homogeneous",
        "wvl_grid": wvl_grid_m,
        "rh_grid": [0.0],
        "refractive_index": fixed_RI,
    }

    # package result
    pop = build_population(pop_cfg)
    bvar = build_variable(name="b_scat", scope="population", var_cfg=var_cfg)
    b_scat_pkg = bvar.compute(pop)
    print('b_scat_pkg',b_scat_pkg)
    if len(b_scat_pkg) == 1:
        b_scat_pkg = b_scat_pkg[0]
    b_scat_pkg = np.asarray(b_scat_pkg, dtype=float)

    # reference result (SI)
    ref = reference_optics_for_population(pop_cfg, var_cfg, wvl_units="m", output_units="m^-1")
    wvl_ref_m = np.asarray(ref["wvl"], dtype=float)
    b_scat_ref = np.asarray(ref["b_scat"], dtype=float)

    # sanity
    assert wvl_ref_m.shape == wvl_grid_m.shape, "wavelength grids misaligned"
    assert b_scat_pkg.shape == b_scat_ref.shape, "shape mismatch between package and reference"
    assert np.all(np.isfinite(b_scat_pkg)) and np.all(np.isfinite(b_scat_ref))
    assert np.all(b_scat_pkg >= -1e-16) and np.all(b_scat_ref >= -1e-16)

    # numeric comparison
    tiny = 1e-16
    abs_diff = np.abs(b_scat_pkg - b_scat_ref)
    rel_err = abs_diff / np.maximum(np.abs(b_scat_ref), tiny)
    rtol = 0.05
    atol = 1e-9

    max_rel = float(np.max(rel_err))
    max_abs = float(np.max(abs_diff))
    if not ((max_rel <= rtol) or (max_abs <= atol)):
        i = int(np.argmax(rel_err))
        raise AssertionError(
            "b_scat mismatch:\n"
            f"  max_rel={max_rel:.3e} at index {i}, lambda={wvl_grid_m[i]:.3e} m\n"
            f"  max_abs={max_abs:.3e}\n"
            f"  pkg={b_scat_pkg[i]:.6e} m^-1, ref={b_scat_ref[i]:.6e} m^-1\n"
        )
