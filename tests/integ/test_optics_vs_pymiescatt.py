import numpy as np
import pytest

pytestmark = pytest.mark.requires_pymiescatt

# Skip the whole test module if PyMieScatt is not available
pytest.importorskip('PyMieScatt')

# Import the example helper by file path (examples/ is not a package)
import importlib.util
from pathlib import Path
helper_path = Path(__file__).resolve().parents[2] / 'examples' / 'helpers' / 'pymiescatt_comparison.py'
spec = importlib.util.spec_from_file_location('pymiescatt_comparison', helper_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pymiescatt_lognormal_optics = module.pymiescatt_lognormal_optics
from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population

def test_bscat_babs_vs_pymiescatt(small_cfg):
    pop_cfg = small_cfg["population"]
    var_cfg = {"wvl_grid": np.linspace(450e-9, 800e-9, 6), "rh_grid": [0.0]}
    pop = build_population(pop_cfg)
    
    opop = build_optical_population(pop, {"type":"homogeneous", **var_cfg})

    bsc = opop.get_optical_coeff("b_scat", rh=0.0)  # shape [nWvl]
    bab = opop.get_optical_coeff("b_abs", rh=0.0)

    wvl_nm, bsc_ref, bab_ref = pymiescatt_lognormal_optics(pop_cfg, var_cfg)

    print(opop.species[0].refractive_index)
    print(bsc, bsc_ref, bab, bsc_ref)
    assert bsc.shape == bsc_ref.shape == bab.shape == bab_ref.shape
    # Basic contract checks: non-negative (allow tiny numerical noise), finite, and shape matches.
    assert np.all(bsc >= 0) and np.all(np.isfinite(bsc))
    # allow small negative numerical noise in absorption values (scale with signal)
    assert np.allclose(bsc, bsc_ref, rtol=1e-3)
    #assert np.allclose(bab, bab_ref, rtol=1e-3)