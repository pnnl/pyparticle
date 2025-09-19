"""Diagnostic comparison: PyParticle vs PyMieScatt b_scat with explicit unit conversions.

This script reads the example config, builds a binned_lognormals population,
computes b_scat via the repository optics builder (homogeneous) and via
PyMieScatt.Mie_Lognormal, and writes a CSV with intermediate unit values to
help debug nm<->m or cm^-3<->m^-3 conversion issues.

Run in the pyparticle-partmc env to include PyMieScatt if available:
  conda run -n pyparticle-partmc python examples/bscat_diagnostic.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
import csv
import numpy as np

from PyParticle import build_population, build_optical_population


def config_to_builder_pop(pop_cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    # Accept builder-style or convert simple example schema
    if isinstance(pop_cfg_in, dict) and 'type' in pop_cfg_in:
        return dict(pop_cfg_in)
    out: Dict[str, Any] = {}
    modes = pop_cfg_in.get('modes', []) if isinstance(pop_cfg_in, dict) else []
    if modes:
        # convert first mode only for this diagnostic
        mode = modes[0]
        out['type'] = 'binned_lognormals'
        out['N'] = [float(mode.get('N', 1000.0))]
        gmd = float(mode.get('gmd', mode.get('GMD', 100e-9)))
        out['GMD'] = [gmd]
        out['GSD'] = [float(mode.get('gsd', mode.get('GSD', 1.6)))]
        spec = mode.get('species', {})
        if spec:
            names = list(spec.keys())
            fracs = [spec[n].get('mass_fraction', 0.0) for n in names]
            total = sum(fracs)
            if total > 0:
                fracs = [f/total for f in fracs]
        else:
            names = ['SO4']
            fracs = [1.0]
        out['aero_spec_names'] = [names]
        out['aero_spec_fracs'] = [fracs]
        out['N_bins'] = pop_cfg_in.get('N_bins', 80)
        out['D_min'] = pop_cfg_in.get('D_min', 10e-9)
        out['D_max'] = pop_cfg_in.get('D_max', 2e-6)
        return out
    # try top-level keys
    if 'GMD' in pop_cfg_in:
        out['type'] = 'binned_lognormals'
        GMD = pop_cfg_in['GMD']
        if isinstance(GMD, (list, tuple)):
            out['GMD'] = [float(GMD[0])]
        else:
            out['GMD'] = [float(GMD)]
        N = pop_cfg_in.get('N', 1000.0)
        out['N'] = [float(N[0]) if isinstance(N, (list, tuple)) else float(N)]
        out['GSD'] = [float(pop_cfg_in.get('GSD', [1.6])[0])]
        out['aero_spec_names'] = pop_cfg_in.get('aero_spec_names', [['SO4']])
        out['aero_spec_fracs'] = pop_cfg_in.get('aero_spec_fracs', [[1.0]])
        out['N_bins'] = pop_cfg_in.get('N_bins', 80)
        return out
    raise ValueError('Unsupported population config format')


def run_diagnostic(cfg_path: str = 'examples/configs/binned_lognormal_bscat_wvl.json'):
    cfg = json.load(open(cfg_path))
    pop_cfg = cfg['population']
    var_cfg = cfg.get('b_scat', cfg.get('optics', {}))

    # Build builder-format config and population
    pop_builder = config_to_builder_pop(pop_cfg)
    pop_builder.setdefault('type', 'binned_lognormals')
    pop = build_population(pop_builder)

    # Optics config for PyParticle (homogeneous)
    ocfg = {
        'type': var_cfg.get('morphology', 'homogeneous'),
        'wvl_grid': list(var_cfg.get('wvl_grid', [550e-9])),
        'rh_grid': list(var_cfg.get('rh_grid', [0.0])),
        'diameter_units': 'm'
    }
    # population-level refractive index
    refr = pop_cfg.get('refractive_index', None)
    if refr is not None and isinstance(refr, (list, tuple)) and len(refr) >= 2:
        ocfg['n_550'] = float(refr[0])
        ocfg['k_550'] = float(refr[1])

    optical_pop = build_optical_population(pop, ocfg)
    # population b_scat from PyParticle (rh dimension x wvl dimension)
    b_scat_pp = optical_pop.get_optical_coeff('b_scat', rh=None, wvl=None)

    # Now compute PyMieScatt Mie_Lognormal results and write comparison CSV
    try:
        import PyMieScatt as PMS
    except Exception as e:
        PMS = None

    # Extract single-mode parameters (first entry)
    GMD_m = pop_builder['GMD'][0]
    GSD = pop_builder['GSD'][0]
    N_m3 = pop_builder['N'][0]
    # convert number conc to cm^-3 for PyMieScatt
    N_cm3 = N_m3 / 1e6

    wvls_m = np.asarray(ocfg['wvl_grid'], dtype=float)
    wvls_nm = (wvls_m * 1e9).tolist()

    out_csv = Path('examples') / 'bscat_diagnostic.csv'
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'wavelength_nm',
            'GMD_m', 'GMD_nm',
            'GSD',
            'N_m3', 'N_cm3',
            'b_scat_pyParticle_m-1',
            'b_scat_pymiescatt_raw', 'pms_raw_units', 'b_scat_pymiescatt_m-1',
            'ratio_pms_over_pyParticle'
        ])

        for iw, wl_nm in enumerate(wvls_nm):
            # PyParticle value at this wavelength (rh=0 index 0)
            try:
                b_pp = float(b_scat_pp[0, iw]) if b_scat_pp.ndim == 2 else float(b_scat_pp[iw])
            except Exception:
                # if b_scat_pp is 1d (wvl only)
                b_pp = float(np.atleast_1d(b_scat_pp)[iw])

            pms_raw = np.nan
            pms_units = ''
            pms_m = np.nan
            if PMS is not None:
                # Call Mie_Lognormal
                out = PMS.Mie_Lognormal(
                    (ocfg.get('n_550', 1.5) + 1j * ocfg.get('k_550', 0.0)),
                    wl_nm,
                    GSD,
                    GMD_m * 1e9,  # GMD in nm
                    N_cm3,
                    lower=10.0,
                    upper=2000.0,
                    asDict=True,
                )
                # Try to extract Bsca from returned dict
                if isinstance(out, dict):
                    if 'Bsca' in out:
                        pms_raw = float(out['Bsca'])
                        pms_units = 'Mm^-1 (assumed)'
                    elif 'Bsca, Mm^-1' in out:
                        pms_raw = float(out['Bsca, Mm^-1'])
                        pms_units = 'Mm^-1'
                    else:
                        # fallback: numeric first element
                        vals = [v for v in out.values() if isinstance(v, (int, float, np.floating))]
                        pms_raw = float(vals[0]) if vals else np.nan
                        pms_units = 'unknown'
                else:
                    # out might be tuple/list: pick first
                    try:
                        pms_raw = float(out[0])
                        pms_units = 'unknown'
                    except Exception:
                        pms_raw = np.nan

                # Convert to m^-1 if value looks like Mm^-1
                if np.isfinite(pms_raw):
                    # Heuristic: PyMieScatt often returns Mm^-1, so divide by 1e6
                    pms_m = pms_raw / 1e6

            ratio = (pms_m / b_pp) if (b_pp != 0 and np.isfinite(pms_m)) else np.nan

            writer.writerow([
                f'{wl_nm:.6f}',
                f'{GMD_m:.6e}', f'{GMD_m*1e9:.6f}',
                f'{GSD:.6f}',
                f'{N_m3:.6e}', f'{N_cm3:.6e}',
                f'{b_pp:.6e}',
                f'{pms_raw:.6e}', pms_units, f'{pms_m:.6e}',
                f'{ratio:.6e}'
            ])

    print('Wrote diagnostic CSV:', out_csv)


if __name__ == "__main__":
    run_diagnostic()
