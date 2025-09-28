from typing import Tuple
import numpy as np
import warnings

try:
    from PyParticle._patch import patch_pymiescatt
    patch_pymiescatt()
    #import PyMieScatt as PMS
    from PyMieScatt import Mie_Lognormal
except Exception as e:
    raise ModuleNotFoundError("Install PyMieScatt to run direct Mie comparison: pip install PyMieScatt") from e

MMINVERSE_TO_MINVERSE = 1e-6  # 1 / (1 Mm) = 1e-6 1/m
M_TO_NM = 1e9                 # meters -> nanometers (for PyMieScatt interface)

# todo: fix this
def pymiescatt_lognormal_optics(pop_cfg, var_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mie optics for a (possibly multi-modal) lognormal aerosol using PyMieScatt,
    returning units that match the rest of PyParticle.

    Parameters
    ----------
    pop_cfg : dict
        Population config. This helper does **not** alter your existing parsing logic
        (refractive index, modes, etc.). It assumes you've already built the parameters
        needed for PyMieScatt internally.
    var_cfg : dict
        Must include 'wvl_grid' **in meters** (SI), as used across PyParticle.

    Returns
    -------
    wvl_nm : np.ndarray
        Wavelengths in **nanometers** (kept for backward compatibility with the notebook,
        which multiplies by 1e-9 before plotting).
    b_scat_m : np.ndarray
        Scattering coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.
    b_abs_m : np.ndarray
        Absorption coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.

    Notes
    -----
    PyMieScatt’s API uses nm for wavelength/diameter and cm⁻³ for concentrations,
    and returns coefficients in Mm⁻¹. See docs. We only standardize the *outputs*
    (to m⁻¹) here so the rest of the package stays SI-consistent.
    """
    # --- your existing logic begins (kept as-is) ---
    # Expect that somewhere below you:
    #   - read var_cfg["wvl_grid"] (meters)
    #   - build wavelength(s) for PyMieScatt in nm
    #   - call PyMieScatt and obtain Bsca/Babs in Mm^-1
    #
    # To keep this patch minimal and risk-free, we don't change how you build inputs.
    # We only enforce output unit conversions right before returning.

    # Handle missing var_cfg gracefully
    if var_cfg is None:
        var_cfg = {}
    
    wvl_m = np.asarray(var_cfg.get("wvl_grid", [550e-9]), dtype=float)
    if wvl_m.ndim != 1:
        wvl_m = wvl_m.reshape(-1)
    wvl_nm = wvl_m * M_TO_NM

    # Helpers
    def _first(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    # Extract modal parameters with safe fallbacks
    gmd = float(_first(pop_cfg.get('GMD', pop_cfg.get('gmd', 0.0))))
    gsd = float(_first(pop_cfg.get('GSD', pop_cfg.get('gsd', 1.6))))
    N0 = float(_first(pop_cfg.get('N', pop_cfg.get('N0', pop_cfg.get('N_tot', 1000.0)))))

    # Convert GMD to nm for PyMieScatt
    gmd_units = pop_cfg.get('GMD_units', 'm')
    if gmd_units == 'm':
        dg_nm = gmd * M_TO_NM
    elif gmd_units in ('nm', 'nanometer', 'nanometers'):
        dg_nm = gmd
    else:
        raise ValueError(f"Unsupported GMD_units: {gmd_units}. Supported: 'm', 'nm'")
    
    # Convert number concentration to cm^-3 for PyMieScatt
    n_units = pop_cfg.get('N_units','m-3')
    if n_units in ('m-3', 'm^-3'):
        N0_cm3 = N0 / 1e6
    elif n_units in ('cm-3', 'cm^-3'):
        N0_cm3 = N0
    else:
        raise ValueError(f"Unsupported N_units: {n_units}. Supported: 'm-3', 'cm-3'")

    # Determine the species-level modifications for the single-species case.
    # species_modifications is expected to be a dict keyed by species name, e.g.
    # {"SO4": {"n_550":1.45, "k_550":0.0}}
    all_spec_mods = pop_cfg.get('species_modifications', {}) or {}
    species_name = None
    # Prefer explicit single-key in species_modifications
    if isinstance(all_spec_mods, dict) and len(all_spec_mods) == 1:
        species_name = list(all_spec_mods.keys())[0]
    # Otherwise, try to infer species from aero_spec_names (list of mode lists)
    if species_name is None:
        aero_spec_names = pop_cfg.get('aero_spec_names', None)
        if aero_spec_names:
            first = aero_spec_names[0]
            if isinstance(first, (list, tuple)):
                species_name = first[0]
            else:
                species_name = first

    # Now pick the per-spec modifications (fall back to empty dict)
    spec_mods = all_spec_mods.get(species_name, {}) if species_name else {}

    ri_real = spec_mods.get('n_550', spec_mods.get('n', 1.5))
    ri_imag = spec_mods.get('k_550', spec_mods.get('k', 0.0))
    if spec_mods.get('alpha_n', 0.0) != 0.0 or spec_mods.get('alpha_k', 0.0) != 0.0:
        warnings.warn("Population-level spectral slope (alpha_n, alpha_k) not supported in PyMieScatt comparison; using n_550/k_550 only.")
    refr = complex(float(ri_real), float(ri_imag))
    print(refr)
    #var_cfg.get('refractive_index', var_cfg.get('m', 1.5 + 0.0j)))
    if isinstance(refr, (list, tuple)) and len(refr) >= 2:
        try:
            n_val = float(refr[0])
            k_val = float(refr[1])
            refr = complex(n_val, k_val)
        except Exception:
            refr = complex(float(refr[0]), 0.0)
    
    # wavelength list for PyMieScatt (nm)
    wl_nm_list = list(wvl_nm)

    # lower/upper cutoffs expected in nm; allow var_cfg to provide lower_nm/upper_nm or lower/upper in nm
    lower = var_cfg.get('lower_nm', var_cfg.get('lower', None))
    upper = var_cfg.get('upper_nm', var_cfg.get('upper', None))
    
    # If the population config provides D_min/D_max, prefer those bounds (convert to nm)
    dmin = pop_cfg.get('D_min', None)
    dmax = pop_cfg.get('D_max', None)
    d_units = pop_cfg.get('GMD_units', pop_cfg.get('D_units', 'm'))
    if dmin is not None and lower is None:
        if d_units == 'm':
            lower = float(dmin) * M_TO_NM
        elif d_units in ('nm', 'nanometer', 'nanometers'):
            lower = float(dmin)
        else:
            lower = float(dmin) * M_TO_NM
    if dmax is not None and upper is None:
        if d_units == 'm':
            upper = float(dmax) * M_TO_NM
        elif d_units in ('nm', 'nanometer', 'nanometers'):
            upper = float(dmax)
        else:
            upper = float(dmax) * M_TO_NM

    # # Fall back to safe defaults if neither provided
    # if lower is None:
    #     lower = dg_nm / 20.0 if gmd > 0 else 1.0
    # if upper is None:
    #     upper = dg_nm * 20.0 if gmd > 0 else 1000.0

    # # Compute builder-derived bounds if GMD/GSD provided and optics did not explicitly set bounds
    # try:
    #     GMD = pop_cfg.get('GMD', None)
    #     GSD = pop_cfg.get('GSD', None)
    #     if GMD is not None and GSD is not None:
    #         gmd_local = float(_first(GMD))
    #         gsd_local = float(_first(GSD))
    #         N_sigmas = float(pop_cfg.get('N_sigmas', 5.0))
    #         import math
    #         dmin_builder = math.exp(math.log(gmd_local) - N_sigmas / 2.0 * math.log(gsd_local))
    #         dmax_builder = math.exp(math.log(gmd_local) + N_sigmas / 2.0 * math.log(gsd_local))
    #         if d_units == 'm':
    #             builder_lower = float(dmin_builder) * M_TO_NM
    #             builder_upper = float(dmax_builder) * M_TO_NM
    #         else:
    #             builder_lower = float(dmin_builder)
    #             builder_upper = float(dmax_builder)

    #         if ('lower' not in var_cfg) and ('lower_nm' not in var_cfg):
    #             lower = builder_lower
    #         if ('upper' not in var_cfg) and ('upper_nm' not in var_cfg):
    #             upper = builder_upper
    # except Exception:
    #     pass

    
    b_scat_Mm1 = []
    b_abs_Mm1 = []
    for wl in wl_nm_list:
        out = Mie_Lognormal(
            refr,
            wl,
            gsd,
            dg_nm,
            N0_cm3,
            lower=lower,
            upper=upper,
            asDict=True,
            numberOfBins=pop_cfg.get('N_bins', 100)
        )

        def _get_key(dct, keys):
            for k in keys:
                if k in dct:
                    return dct[k]
            return None

        if isinstance(out, dict):
            bsca_raw = _get_key(out, ['Bsca', 'Bsca, Mm^-1', 'Bsca, Mm^-1', 'Bsca (Mm^-1)'])
            babs_raw = _get_key(out, ['Babs', 'Babs, Mm^-1', 'Babs (Mm^-1)'])
        else:
            try:
                bsca_raw = out[0]
            except Exception:
                bsca_raw = 0.0
            try:
                babs_raw = out[1]
            except Exception:
                babs_raw = 0.0

        bsca_raw = float(bsca_raw) if bsca_raw is not None else 0.0
        babs_raw = float(babs_raw) if babs_raw is not None else 0.0

        b_scat_Mm1.append(bsca_raw)
        b_abs_Mm1.append(babs_raw)

    b_scat_Mm1 = np.asarray(b_scat_Mm1, dtype=float)
    b_abs_Mm1 = np.asarray(b_abs_Mm1, dtype=float)

    # Convert to m^-1
    b_scat_m = b_scat_Mm1 * MMINVERSE_TO_MINVERSE
    b_abs_m = b_abs_Mm1 * MMINVERSE_TO_MINVERSE

    return np.asarray(wl_nm_list), b_scat_m, b_abs_m
