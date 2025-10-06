import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:.*alpha.*")  # if helper used to warn

def mieq_cs(m, wl_nm, D_nm, PMS):
    out = PMS.MieQ(m, wl_nm, D_nm, asDict=True, asCrossSection=True)
    return out["Cext"]*1e-18, out["Csca"]*1e-18, out["Cabs"]*1e-18  # nm²→m²

def test_single_sphere_cross_section_parity():
    PMS = pytest.importorskip("PyMieScatt")
    from PyParticle.species.registry import retrieve_one_species
    from PyParticle.optics.refractive_index import build_refractive_index

    wl_nm = 550.0
    wl_m  = wl_nm*1e-9
    D_nm  = 100.0

    # pick a species present in your example config (adjust key as needed)
    key = "SO4"
    spec = retrieve_one_species(key)
    mods = {"n_550":1.52, "k_550":0.0, "alpha_n":0.0, "alpha_k":0.0}
    build_refractive_index(spec, np.array([wl_m]), modifications=mods)

    m_pp = complex(spec.refractive_index.real_ri_fun(wl_m),
                   spec.refractive_index.imag_ri_fun(wl_m))
    m_550 = complex(mods["n_550"], mods["k_550"])

    Cext_pp, Csca_pp, Cabs_pp = mieq_cs(m_pp, wl_nm, D_nm, PMS)
    Cext_55, Csca_55, Cabs_55 = mieq_cs(m_550, wl_nm, D_nm, PMS)

    # If alphas are zero, these must match closely
    for a,b in [(Cext_pp,Cext_55),(Csca_pp,Csca_55),(Cabs_pp,Cabs_55)]:
        rel = 0.0 if a==0 else abs(a-b)/max(abs(a),1e-30)
        assert rel < 1e-3


import numpy as np
import pytest

def test_core_shell_q_vs_cross_section_routes_match():
    PMS = pytest.importorskip("PyMieScatt")
    mcore  = complex(1.8, 0.05)
    mshell = complex(1.5, 0.0)
    wl_nm = 550.0
    Dcore_nm  = 80.0
    Dshell_nm = 120.0
    Dshell_m  = Dshell_nm * 1e-9

    out_cs = PMS.MieQCoreShell(mcore, mshell, wl_nm, Dcore_nm, Dshell_nm,
                               asDict=True, asCrossSection=True)
    cx_cs = out_cs["Cext"]*1e-18  # nm²→m²

    out_q = PMS.MieQCoreShell(mcore, mshell, wl_nm, Dcore_nm, Dshell_nm,
                              asDict=True, asCrossSection=False)
    area_m2 = np.pi*(Dshell_m/2.0)**2
    cx_q = out_q["Qext"]*area_m2

    rel = abs(cx_q - cx_cs)/max(abs(cx_cs),1e-30)
    assert rel < 1e-3
