import numpy as np
from typing import Dict, Any
import warnings

def pymiescatt_lognormal_optics(pop_cfg: Dict[str, Any], optics_cfg: Dict[str, Any]):
    try:
        import PyMieScatt as PMS
    except Exception as e:
        raise ModuleNotFoundError("Install PyMieScatt to run direct Mie comparison: pip install PyMieScatt") from e

    # Prefer explicit unit keys, but assume SI when they are absent.
    # This keeps backward compatibility while encouraging explicit units.
    required_pop_units = ("GMD_units", "N_units")
    missing = [k for k in required_pop_units if k not in pop_cfg]
    if missing:
        warnings.warn(
            f"Population config missing unit keys {missing}; assuming SI defaults (GMD in meters, N in m-3)."
            " Add explicit 'GMD_units' and 'N_units' to remove this warning.",
            UserWarning,
        )
        pop_cfg.setdefault('GMD_units', 'm')
        pop_cfg.setdefault('N_units', 'm-3')
    if 'wvl_units' not in optics_cfg:
        warnings.warn(
            "Optics config missing 'wvl_units'; assuming meters (m). Add explicit 'wvl_units' to remove this warning.",
            UserWarning,
        )
        # allow mutation for local convenience
        optics_cfg.setdefault('wvl_units', 'm')

    # Extract scalar values (support list or scalar entries)
    def _first(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    gmd = float(_first(pop_cfg.get('GMD', pop_cfg.get('gmd', 0.0))))
    gsd = float(_first(pop_cfg.get('GSD', pop_cfg.get('gsd', 1.6))))
    N0 = float(_first(pop_cfg.get('N', pop_cfg.get('N0', pop_cfg.get('N_tot', 1000.0)))))

    # Convert GMD to nm for PyMieScatt
    gmd_units = pop_cfg['GMD_units']
    if gmd_units == 'm':
        dg_nm = gmd * 1e9
    elif gmd_units in ('nm', 'nanometer', 'nanometers'):
        dg_nm = gmd
    else:
        raise ValueError(f"Unsupported GMD_units: {gmd_units}. Supported: 'm', 'nm'")

    # Convert number concentration to cm^-3 for PyMieScatt
    n_units = pop_cfg['N_units']
    if n_units == 'm-3':
        N0_cm3 = N0 / 1e6
    elif n_units in ('cm-3', 'cm^-3'):
        N0_cm3 = N0
    else:
        raise ValueError(f"Unsupported N_units: {n_units}. Supported: 'm-3', 'cm-3'")

    # Wavelengths: convert to nm
    wvl_units = optics_cfg['wvl_units']
    wvl_grid = list(optics_cfg.get('wvl_grid', [550e-9]))
    if wvl_units == 'm':
        wl_nm_list = [float(w) * 1e9 for w in wvl_grid]
    elif wvl_units == 'nm':
        wl_nm_list = [float(w) for w in wvl_grid]
    else:
        raise ValueError(f"Unsupported wvl_units: {wvl_units}. Supported: 'm', 'nm'")

    # refractive index: allow population-level refractive_index key or optics_cfg 'refractive_index'
    refr = pop_cfg.get('refractive_index', optics_cfg.get('refractive_index', optics_cfg.get('m', 1.5 + 0.0j)))
    # If refr is provided as [n, k] convert to complex(n, k)
    if isinstance(refr, (list, tuple)) and len(refr) >= 2:
        try:
            n_val = float(refr[0])
            k_val = float(refr[1])
            refr = complex(n_val, k_val)
        except Exception:
            # fall back to default scalar
            refr = complex(float(refr[0]), 0.0)

    # lower/upper cutoffs expected in nm; allow optics_cfg to provide lower_nm/upper_nm or lower/upper in nm
    lower = optics_cfg.get('lower_nm', optics_cfg.get('lower', None))
    upper = optics_cfg.get('upper_nm', optics_cfg.get('upper', None))

    # If the population config provides D_min/D_max, prefer those bounds (convert to nm
    # using the population diameter units if present). This aligns the analytic
    # PyMieScatt integration limits with the discrete bins used by the builder.
    try:
        dmin = pop_cfg.get('D_min', None)
        dmax = pop_cfg.get('D_max', None)
    except Exception:
        dmin = None
        dmax = None

    # Determine units for diameter values; fall back to meters for backward compatibility
    d_units = pop_cfg.get('GMD_units', pop_cfg.get('D_units', 'm'))
    if dmin is not None and lower is None:
        if d_units == 'm':
            lower = float(dmin) * 1e9
        elif d_units in ('nm', 'nanometer', 'nanometers'):
            lower = float(dmin)
        else:
            # unknown units: assume meters
            lower = float(dmin) * 1e9
    if dmax is not None and upper is None:
        if d_units == 'm':
            upper = float(dmax) * 1e9
        elif d_units in ('nm', 'nanometer', 'nanometers'):
            upper = float(dmax)
        else:
            upper = float(dmax) * 1e9

    # Fall back to safe defaults if neither optics_cfg nor pop_cfg provided bounds
    if lower is None:
        lower = 10.0
    if upper is None:
        upper = 2000.0
    
    # Compute the builder's implicit bin range (used by binned_lognormals) from
    # GMD/GSD if available, and prefer those bounds unless optics_cfg explicitly
    # supplied lower/upper. This aligns the analytic PyMieScatt integration with
    # the discrete binning used by the population builder.
    try:
        GMD = pop_cfg.get('GMD', None)
        GSD = pop_cfg.get('GSD', None)
        if GMD is not None and GSD is not None:
            # support list or scalar
            def _first(x):
                if isinstance(x, (list, tuple)):
                    return x[0]
                return x

            gmd = float(_first(GMD))
            gsd = float(_first(GSD))
            N_sigmas = float(pop_cfg.get('N_sigmas', 5.0))
            import math
            dmin_builder = math.exp(math.log(gmd) - N_sigmas / 2.0 * math.log(gsd))
            dmax_builder = math.exp(math.log(gmd) + N_sigmas / 2.0 * math.log(gsd))
            # convert to nm if meters
            if d_units == 'm':
                builder_lower = float(dmin_builder) * 1e9
                builder_upper = float(dmax_builder) * 1e9
            else:
                builder_lower = float(dmin_builder)
                builder_upper = float(dmax_builder)

            # Only override if optics did not explicitly set lower/upper
            # (we treat None as unset). If user provided explicit bounds, keep them.
            if optics_cfg.get('lower') is None and optics_cfg.get('lower_nm') is None and ('lower' not in optics_cfg and 'lower_nm' not in optics_cfg):
                lower = builder_lower
            if optics_cfg.get('upper') is None and optics_cfg.get('upper_nm') is None and ('upper' not in optics_cfg and 'upper_nm' not in optics_cfg):
                upper = builder_upper
    except Exception:
        # if anything goes wrong, keep previously-determined lower/upper
        pass

    b_scat_m = []
    b_abs_m = []
    for wl in wl_nm_list:
        out = PMS.Mie_Lognormal(
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
        # Extract Bsca and Babs (PyMieScatt ensemble returns Bsca/Babs commonly in Mm^-1)
        def _get_key(dct, keys):
            for k in keys:
                if k in dct:
                    return dct[k]
            return None

        if isinstance(out, dict):
            bsca_raw = _get_key(out, ['Bsca', 'Bsca, Mm^-1', 'Bsca (Mm^-1)'])
            babs_raw = _get_key(out, ['Babs', 'Babs, Mm^-1', 'Babs (Mm^-1)'])
        else:
            # fallback: try sequence positions (PMS may return tuple)
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

        # PyMieScatt ensemble outputs are conventionally in Mm^-1 (1e6 m^-1). Convert deterministically.
        b_scat_m.append(bsca_raw / 1e6)
        b_abs_m.append(babs_raw / 1e6)

    return np.asarray(wl_nm_list), np.asarray(b_scat_m), np.asarray(b_abs_m)
