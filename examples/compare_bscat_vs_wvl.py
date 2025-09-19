"""Compare PyParticle `core_shell` optical result to a direct PyMieScatt lognormal calculation.

See repo examples for usage and the config under examples/configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import warnings

from PyParticle import build_population, build_optical_population
from PyParticle import make_particle

def pymiescatt_lognormal_b_scat(pop_cfg: Dict[str, Any], optics_cfg: Dict[str, Any], rh: float = 0.0):
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/configs/binned_lognormal_bscat_wvl.json")
    args = p.parse_args(argv)

    cfg = json.load(open(args.config))
    pop_cfg = cfg["population"]
    var_cfg = cfg.get("b_scat", cfg.get("optics", {}))

    # Convert example 'population' config schema to a builder-friendly format
    def config_to_builder_pop(pop_cfg_in: Dict[str, Any]) -> Dict[str, Any]:
        # If already appears to be a builder config (has 'type'), return copy
        if isinstance(pop_cfg_in, dict) and 'type' in pop_cfg_in:
            return dict(pop_cfg_in)
        # Do not support the shorthand 'modes' dictionary in this example. Require
        # a builder-style configuration (with 'type' and parallel lists such as
        # 'N', 'GMD', 'GSD', 'aero_spec_names', 'aero_spec_fracs'). This keeps the
        # example unambiguous and avoids implicit unit assumptions.
        raise ValueError(
            "Unsupported population config format: expected builder-style config with a 'type' key."
            " Remove any 'modes' shorthand and provide keys like 'N', 'GMD', 'GSD',"
            " 'aero_spec_names', and 'aero_spec_fracs' as parallel lists (see examples/configs/binned_lognormal_bscat_wvl.json)."
        )

    pop_cfg_builder = config_to_builder_pop(pop_cfg)
    pop_cfg_builder.setdefault("type", "binned_lognormals")
    pop = build_population(pop_cfg_builder)

    # Build optics config from variable cfg and population refractive index
    opt_cfg_builder = dict(var_cfg)
    opt_cfg_builder.setdefault("type", var_cfg.get("morphology", "homogeneous"))
    opt_cfg_builder["diameter_units"] = "m"
    # Force zero dispersion so refractive index stays constant across wavelength
    opt_cfg_builder.setdefault("alpha_n", 0.0)
    opt_cfg_builder.setdefault("alpha_k", 0.0)
    # If population defines a refractive index (e.g., [n, k]) use it
    refr = pop_cfg.get('refractive_index', pop_cfg.get('refractive_indices', None))
    if refr is not None:
        if isinstance(refr, (list, tuple)) and len(refr) >= 2:
            opt_cfg_builder['n_550'] = float(refr[0])
            opt_cfg_builder['k_550'] = float(refr[1])
        else:
            # allow scalar real index
            opt_cfg_builder['n_550'] = float(refr)

    opt_pop = build_optical_population(pop, opt_cfg_builder)

    # Extract b_scat vs wavelength at RH=0 using OpticalPopulation API
    wl = np.asarray(opt_cfg_builder.get("wvl_grid", [550e-9]))

    # Diagnostic: print refractive index used per wavelength to confirm no dispersion
    print("Refractive indices used by PyParticle:")
    for i, w in enumerate(wl):
        try:
            n_val = opt_pop.n_complex[i].real
            k_val = opt_pop.n_complex[i].imag
            print(f"{w*1e9:.0f} nm: n={n_val:.3f}, k={k_val:.3f}")
        except Exception:
            # if opt_pop.n_complex not available or shaped differently, skip gracefully
            pass
    rh_val = float(opt_cfg_builder.get("rh_grid", [0.0])[0])
    # get_optical_coeff returns array across wavelengths when wvl is None
    b_scat_pp = opt_pop.get_optical_coeff("b_scat", rh=rh_val, wvl=None)
    
    try:
        # Pass population cfg (which contains refractive index) and variable cfg
        # Wrapper returns (wl_nm, b_scat_m, b_abs_m)
        wl_p, b_scat_pms_m, b_abs_pms_m = pymiescatt_lognormal_b_scat(pop_cfg, var_cfg, rh=0.0)
    except Exception as e:
        print("PyMieScatt comparison failed:", e)
        wl_p, b_scat_pms_m, b_abs_pms_m = (wl * 1e9), np.zeros_like(wl), np.zeros_like(wl)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl * 1e9, b_scat_pp, marker="o", label="PyParticle core_shell b_scat")
    # wl_p returned by wrapper is in nm already
    ax.plot(wl_p, b_scat_pms_m, marker="x", linestyle="--", label="PyMieScatt lognormal b_scat")
    # compute percent difference (avoid divide-by-zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = 100.0 * (b_scat_pp - b_scat_pms_m) / np.where(b_scat_pms_m != 0, b_scat_pms_m, np.nan)
    # annotate max absolute percent difference
    if np.any(np.isfinite(pct_diff)):
        max_abs_pct = np.nanmax(np.abs(pct_diff))
        ax.annotate(f"max |% diff| = {max_abs_pct:.1f}%", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                    verticalalignment='top')
    else:
        ax.annotate("percent diff undefined (zero reference)", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                    verticalalignment='top')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("b_scat (m^-1)")
    ax.set_title("b_scat vs wavelength at RH=0")
    ax.grid(True)
    ax.legend()

    out = Path("examples") / "out_bscat_vs_wvl.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out}")
    # Write numeric comparison CSV
    import csv
    csv_out = Path("examples") / "bscat_comparison.csv"
    with open(csv_out, "w", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["wavelength_nm", "b_scat_pyParticle", "b_scat_pymiescatt_m-1", "b_abs_pymiescatt_m-1", "pct_diff_b_scat"])
        for w, a, b, babs, p in zip(wl * 1e9, b_scat_pp, b_scat_pms_m, b_abs_pms_m, pct_diff):
            writer.writerow([f"{w:.2f}", f"{a:.6e}", f"{b:.6e}", f"{babs:.6e}", f"{p:.6e}"])
    print(f"Wrote: {csv_out}")

    # --- b_abs vs wavelength comparison plot ---
    try:
        b_abs_pp = opt_pop.get_optical_coeff("b_abs", rh=rh_val, wvl=None)
    except Exception:
        b_abs_pp = np.zeros_like(b_scat_pp)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(wl * 1e9, b_abs_pp, marker="o", label="PyParticle core_shell b_abs")
    ax2.plot(wl_p, b_abs_pms_m, marker="x", linestyle="--", label="PyMieScatt lognormal b_abs")
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff_abs = 100.0 * (b_abs_pp - b_abs_pms_m) / np.where(b_abs_pms_m != 0, b_abs_pms_m, np.nan)
    if np.any(np.isfinite(pct_diff_abs)):
        max_abs_pct_abs = np.nanmax(np.abs(pct_diff_abs))
        ax2.annotate(f"max |% diff| = {max_abs_pct_abs:.1f}%", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                     verticalalignment='top')
    else:
        ax2.annotate("percent diff undefined (zero reference)", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                     verticalalignment='top')
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("b_abs (m^-1)")
    ax2.set_title("b_abs vs wavelength at RH=0")
    ax2.grid(True)
    ax2.legend()
    out2 = Path("examples") / "out_babs_vs_wvl.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=180)
    print(f"Wrote: {out2}")

    # Write absorption comparison CSV
    csv_abs = Path("examples") / "babs_comparison.csv"
    with open(csv_abs, "w", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["wavelength_nm", "b_abs_pyParticle", "b_abs_pymiescatt_m-1", "pct_diff_b_abs"])
        for w, a_abs, b_abs_val, p_abs in zip(wl * 1e9, b_abs_pp, b_abs_pms_m, pct_diff_abs):
            writer.writerow([f"{w:.2f}", f"{a_abs:.6e}", f"{b_abs_val:.6e}", f"{p_abs:.6e}"])
    print(f"Wrote: {csv_abs}")

    # --- Make a figure of b_scat vs RH for a homogeneous sphere ---
    def plot_b_scat_vs_rh_homogeneous():
        # Determine diameter and number from the builder config
        # prefer pop_cfg_builder (which was translated to builder format)
        try:
            D = float(pop_cfg_builder.get('GMD', [200e-9])[0])
        except Exception:
            D = float(pop_cfg_builder.get('GMD', 200e-9))
        try:
            N = float(pop_cfg_builder.get('N', [1000.0])[0])
        except Exception:
            N = float(pop_cfg_builder.get('N', 1000.0))
        # Use binned lognormal builder config (created earlier) to build a
        # binned_lognormal population for the RH sweep. Ensure the builder
        # configuration uses a single SO4 mode.
        try:
            binned_cfg = pop_cfg_builder.copy()
            binned_cfg.setdefault('type', 'binned_lognormals')
            # Force single-species SO4 with full mass fraction if not present
            if 'aero_spec_names' in binned_cfg:
                # already aligned
                pass
            else:
                binned_cfg['aero_spec_names'] = [['SO4']]
                binned_cfg['aero_spec_fracs'] = [[1.0]]
            binned_pop = build_population(binned_cfg)
        except Exception as e:
            print('Failed to build binned_lognormal population for RH sweep:', e)
            return
        rh_grid = np.linspace(0.0, 0.95, 10)
        wvl_grid = np.asarray(var_cfg.get('wvl_grid', [550e-9]))
        # Map refractive index keys if present
        ri = pop_cfg.get('refractive_index_core', pop_cfg.get('refractive_index', pop_cfg.get('refractive_indices', None)))
        n_550 = None
        k_550 = None
        if isinstance(ri, (list, tuple)) and len(ri) >= 2:
            n_550 = float(ri[0])
            k_550 = float(ri[1])

        ocfg = {
            'type': 'homogeneous',
            'rh_grid': list(rh_grid),
            'wvl_grid': list(wvl_grid),
            'diameter_units': 'm',
        }
        if n_550 is not None:
            ocfg['n_550'] = n_550
        if k_550 is not None:
            ocfg['k_550'] = k_550

        optical_pop = build_optical_population(binned_pop, ocfg)
        # Get scattering coefficient per RH (first wavelength)
        try:
            b_scat_arr = optical_pop.get_optical_coeff('b_scat', rh=None, wvl=None)
        except Exception:
            # Fallback: sum per-particle cross sections
            bs = np.zeros((len(rh_grid), len(wvl_grid)))
            for i, particle in enumerate(optical_pop.particles):
                Csca = particle.get_cross_section('b_scat')
                bs += np.asarray(Csca) * optical_pop.num_concs[i]
            b_scat_arr = bs

        fig, ax = plt.subplots(figsize=(6, 4))
        # If b_scat_arr is 2D [rh, wvl], plot first wavelength vs RH
        ax.plot(rh_grid, b_scat_arr[:, 0], marker='o')
        ax.set_xlabel('RH')
        ax.set_ylabel('b_scat (m^-1)')
        ax.set_title('b_scat vs RH (homogeneous sphere)')
        ax.grid(True)
        out_rh = Path('examples') / 'out_bscat_vs_rh.png'
        fig.tight_layout()
        fig.savefig(out_rh, dpi=180)
        print(f'Wrote: {out_rh}')

        # Also compute/plot b_abs vs RH if available
        try:
            b_abs_arr = optical_pop.get_optical_coeff('b_abs', rh=None, wvl=None)
            fig_ab, ax_ab = plt.subplots(figsize=(6, 4))
            ax_ab.plot(rh_grid, b_abs_arr[:, 0], marker='o')
            ax_ab.set_xlabel('RH')
            ax_ab.set_ylabel('b_abs (m^-1)')
            ax_ab.set_title('b_abs vs RH (homogeneous sphere)')
            ax_ab.grid(True)
            out_ab_rh = Path('examples') / 'out_babs_vs_rh.png'
            fig_ab.tight_layout()
            fig_ab.savefig(out_ab_rh, dpi=180)
            print(f'Wrote: {out_ab_rh}')
        except Exception:
            # if unavailable, skip silently
            pass

    plot_b_scat_vs_rh_homogeneous()


if __name__ == "__main__":
    main()
