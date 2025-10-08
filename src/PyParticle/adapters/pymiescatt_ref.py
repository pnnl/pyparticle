"""PyMieScatt reference adapter for tests and examples.

Provides two convenience functions used by notebooks/tests:
- pymiescatt_lognormal_optics(pop_cfg, var_cfg) -> (wvl_nm, b_scat_m, b_abs_m)
- pymiescatt_core_shell_optics(pop_cfg, var_cfg) -> (wvl_nm, b_scat_m, b_abs_m)

This module is deliberately small and raises a clear ImportError if PyMieScatt
is not installed in the environment.
"""
# from __future__ import annotations
# import numpy as np
# from typing import Tuple, Sequence

# try:
#     import PyMieScatt as pms
# except Exception as e:  # pragma: no cover - handled at runtime in tests/examples
#     pms = None
#     _IMPORT_ERR = e


# def _require_pms():
#     if pms is None:
#         raise ImportError("PyMieScatt is required by PyParticle.adapters.pymiescatt_ref") from _IMPORT_ERR


# def _m_to_nm(x: Sequence[float]) -> np.ndarray:
#     return np.asarray(x, dtype=float) * 1e9


# def _N_m3_to_cm3(x: float) -> float:
#     return float(x) * 1e-6


# def pymiescatt_lognormal_optics(pop_cfg: dict, var_cfg: dict) -> Tuple[list, np.ndarray, np.ndarray]:
#     """Compute reference Mie optics for a single-mode lognormal using PyMieScatt.

#     Returns wavelengths in nm and coefficients in m^-1 to match notebook expectations.
#     This implementation follows the signature used in tests/examples and is
#     intentionally conservative (single-mode path).
#     """
#     _require_pms()

#     # extract basic pop fields (assume single-mode for examples/tests)
#     N = float(pop_cfg["N"][0])
#     GMD_m = float(pop_cfg["GMD"][0])
#     GSD = float(pop_cfg["GSD"][0])
#     n_bins = int(pop_cfg.get("N_bins", 200))

#     # wavelengths expected in var_cfg as meters -> convert to nm for PyMieScatt
#     wvl_m = var_cfg.get("wvl_grid",[550e-9]);
#     wvl_nm = _m_to_nm(wvl_m)

#     # number concentration conversion to cm^-3 (PyMieScatt expects cm^-3)
#     N_cm3 = _N_m3_to_cm3(N)

#     # species refractive index (if provided) — pick first species_mod if present
#     mods = pop_cfg.get("species_modifications") or {}
#     spec = mods[next(iter(mods.keys()))] if len(mods) > 0 else {}
#     n550 = float(spec.get("n_550", 1.55))
#     k550 = float(spec.get("k_550", 0.0))

#     # simple wavelength dependence helpers using alpha_n/alpha_k if present
#     alpha_n = float(spec.get("alpha_n", 0.0))
#     alpha_k = float(spec.get("alpha_k", 0.0))

#     def complex_m(lam_nm):
#         lam_m = lam_nm * 1e-9
#         n = n550 * (lam_m / 550e-9) ** alpha_n
#         k = k550 * (lam_m / 550e-9) ** alpha_k
#         return complex(n, k)

#     b_scat = []
#     b_abs = []
#     for lam in wvl_nm:
#         m = complex_m(lam)
#         # Use Mie_Lognormal (returns Mm^-1 for cm^-3 input) — convert to m^-1
#         Bext, Bsca, Babs, _g = pms.Mie_Lognormal(
#             m, float(lam),
#             GMD=float(GMD_m * 1e9),
#             GSD=GSD,
#             lower=None, upper=None,
#             numberOfBins=n_bins,
#             numberConcentration=N_cm3,
#         )
#         # Bsca/Babs are in Mm^-1 -> convert to m^-1
#         b_scat.append(float(Bsca) * 1e-6)
#         b_abs.append(float(Babs) * 1e-6)

#     return list(map(float, wvl_nm.tolist() if hasattr(wvl_nm, 'tolist') else list(wvl_nm))), np.asarray(b_scat), np.asarray(b_abs)


from __future__ import annotations
from typing import Tuple, Sequence
import numpy as np

try:
    import PyMieScatt as pms
except Exception as e:  # handled at runtime by raising clearer error
    pms = None
    _PMS_IMPORT_ERR = e


def _require_pms():
    if pms is None:
        raise ImportError(
            "PyMieScatt is required by PyParticle.adapters.pymiescatt_ref. "
            "Install it in your environment, e.g.:  python -m pip install PyMieScatt"
        ) from _PMS_IMPORT_ERR


def _m_to_nm(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1e9


def _N_m3_to_cm3(x: float) -> float:
    return float(x) * 1e-6


def pymiescatt_lognormal_optics(pop_cfg: dict, var_cfg: dict) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Reference optics for a *single-mode* lognormal, using PyMieScatt's Mie_Lognormal.

    Returns:
      wvl_nm: list of wavelengths [nm]
      b_scat_m, b_abs_m: arrays [m^-1]
    """
    _require_pms()

    # ----- population inputs (single mode) -----
    # Required keys: N, GMD, GSD; optional: N_bins, D_min, D_max, N_sigmas
    N_m3  = float(pop_cfg["N"][0])
    GMD_m = float(pop_cfg["GMD"][0])
    GSD   = float(pop_cfg["GSD"][0])
    n_bins = int(pop_cfg.get("N_bins", 200))

    # Wavelengths (meters -> nm for PyMieScatt)
    wvl_m  = var_cfg.get("wvl_grid",[550e-9])
    wvl_nm = _m_to_nm(wvl_m)

    # Concentration: m^-3 -> cm^-3
    N_cm3 = _N_m3_to_cm3(N_m3)

    # Diameter bounds (nm): respect explicit D_min/D_max if given; else ±N_sigmas about ln(GMD)
    if ("D_min" in pop_cfg) and ("D_max" in pop_cfg) and (pop_cfg["D_min"] is not None) and (pop_cfg["D_max"] is not None):
        Dmin_nm = float(pop_cfg["D_min"]) * 1e9
        Dmax_nm = float(pop_cfg["D_max"]) * 1e9
    else:
        N_sigmas = float(pop_cfg.get("N_sigmas", 5.0))
        lnG = np.log(GMD_m)
        lnS = np.log(GSD)
        Dmin_nm = float(np.exp(lnG - 0.5 * N_sigmas * lnS) * 1e9)
        Dmax_nm = float(np.exp(lnG + 0.5 * N_sigmas * lnS) * 1e9)
    if not (Dmin_nm > 0.0 and Dmax_nm > Dmin_nm):
        raise ValueError(f"Invalid diameter bounds: D_min={Dmin_nm} nm, D_max={Dmax_nm} nm")

    # ----- refractive index vs wavelength -----
    # Expect a single 'species_modifications' entry with n_550/k_550 and optional alpha_n/alpha_k
    mods = pop_cfg.get("species_modifications", {})
    if len(mods) > 0:
        first = next(iter(mods.values()))
        n550  = float(first.get("n_550", 1.55))
        k550  = float(first.get("k_550", 0.0))
        alpha_n = float(first.get("alpha_n", 0.0))
        alpha_k = float(first.get("alpha_k", 0.0))
    else:
        n550, k550, alpha_n, alpha_k = 1.55, 0.0, 0.0, 0.0

    def m_at_nm(lam_nm: float) -> complex:
        lam_m = lam_nm * 1e-9
        n = n550 * (lam_m / 550e-9) ** alpha_n
        k = k550 * (lam_m / 550e-9) ** alpha_k
        return complex(n, k)

    GMD_nm = GMD_m * 1e9

    # ----- main loop over wavelengths -----
    b_scat = []
    b_abs  = []
    for lam_nm in wvl_nm:
        m = m_at_nm(float(lam_nm))
        # NOTE: PyMieScatt expects nm units for wavelength and diameters; numberConcentration in cm^-3.
        # It returns B* in Mm^-1 -> convert to m^-1 with 1e-6.
        # Bext, Bsca, Babs, _g = pms.Mie_Lognormal(
        #     m, float(lam_nm),
        #     GMD=GMD_nm,
        #     GSD=GSD,
        #     lower=Dmin_nm, upper=Dmax_nm,         # <-- respect explicit bounds if provided
        #     numberOfBins=n_bins,
        #     numberConcentration=N_cm3,
        # )

        #Bext, Bsca, Babs, _g
        Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio = pms.Mie_Lognormal(
            m, float(lam_nm),
            geoStdDev=GSD,
            geoMean=GMD_nm,
            numberOfParticles=N_cm3,
            nMedium=1.0, 
            numberOfBins=n_bins,
            lower=Dmin_nm,upper=Dmax_nm,
            asDict=False)

        b_scat.append(float(Bsca) * 1e-6)
        b_abs.append(float(Babs) * 1e-6)

    # Return nm wavelengths (for plotting) and SI coefficients
    return [float(x) for x in wvl_nm], np.asarray(b_scat), np.asarray(b_abs)


def pymiescatt_core_shell_optics(pop_cfg: dict, var_cfg: dict) -> Tuple[list, np.ndarray, np.ndarray]:
    """Lightweight core–shell reference using PyMieScatt's MieQCoreShell per bin.

    This function favors clarity over speed — intended for small demo grids and tests.
    """
    _require_pms()

    # reuse the same binning strategy as the lognormal adapter: create mids from GMD/GSD
    GMD_m = float(pop_cfg["GMD"][0])
    GSD = float(pop_cfg["GSD"][0])
    n_bins = int(pop_cfg.get("N_bins", 200))
    N = float(pop_cfg["N"][0])
    N_cm3 = _N_m3_to_cm3(N)

    # diameter bounds: use D_min/D_max if provided, else +/- 5 sigma heuristic
    if pop_cfg.get("D_min") is not None and pop_cfg.get("D_max") is not None:
        dmin_nm = float(pop_cfg["D_min"]) * 1e9
        dmax_nm = float(pop_cfg["D_max"]) * 1e9
    else:
        N_sigmas = float(pop_cfg.get("N_sigmas", 5))
        dmin_nm = float(np.exp(np.log(GMD_m) - 0.5 * N_sigmas * np.log(GSD)) * 1e9)
        dmax_nm = float(np.exp(np.log(GMD_m) + 0.5 * N_sigmas * np.log(GSD)) * 1e9)

    edges = np.logspace(np.log10(dmin_nm), np.log10(dmax_nm), n_bins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])

    # simple core fraction (can be overridden by pop_cfg['core_fraction'])
    core_frac = float(pop_cfg.get("core_fraction", 0.2))

    # wavelengths
    wvl_m = var_cfg.get("wvl_grid", [550e-9])
    wvl_nm = _m_to_nm(wvl_m)

    bsca_res = []
    babs_res = []
    for lam in wvl_nm:
        # pick RIs from species_modifications if present
        mods = pop_cfg.get("species_modifications", {})
        # default core/shell n/k
        core_n = float(mods.get("BC", {}).get("n_550", 1.85))
        core_k = float(mods.get("BC", {}).get("k_550", 0.8))
        shell_name = next((k for k in mods.keys() if k != "BC"), None)
        shell_n = float(mods.get(shell_name, {}).get("n_550", 1.55))
        shell_k = float(mods.get(shell_name, {}).get("k_550", 0.0))

        # per-bin cross sections
        Csca = np.zeros_like(mids)
        Cabs = np.zeros_like(mids)
        for i, Dnm in enumerate(mids):
            Dcore = Dnm * (core_frac ** (1/3.0))
            out = pms.MieQCoreShell(complex(core_n, core_k), complex(shell_n, shell_k), float(lam), float(Dcore), float(Dnm), asDict=True, asCrossSection=False)
            # area conversion: MieQCoreShell returns Q; convert to cross-section via geometric area (nm^2)
            area_nm2 = np.pi * (0.5 * Dnm) ** 2
            Csca[i] = out["Qsca"] * area_nm2 * 1e-18  # -> m^2
            Cabs[i] = out["Qabs"] * area_nm2 * 1e-18
        
        # number per bin: approximate lognormal PDF weight
        pdf = (1.0 / (np.log(GSD) * np.sqrt(2 * np.pi))) * np.exp(- (np.log(mids) - np.log(GMD_m * 1e9)) ** 2 / (2 * (np.log(GSD) ** 2)))
        dln = np.log(edges[1:]) - np.log(edges[:-1])
        N_bin_cm3 = (pdf * dln) * N_cm3
        N_bin_m3 = N_bin_cm3 * 1e6

        bsca_res.append(float(np.sum(Csca * N_bin_m3)))
        babs_res.append(float(np.sum(Cabs * N_bin_m3)))

    return list(map(float, wvl_nm.tolist() if hasattr(wvl_nm, 'tolist') else list(wvl_nm))), np.asarray(bsca_res), np.asarray(babs_res)
