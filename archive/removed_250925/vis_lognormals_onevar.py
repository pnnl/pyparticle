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
from PyParticle.viz_lessOld import plot_lines

def plot_one_line(varname='b_scat',
                  base_config_path="examples/configs/binned_lognormal_bscat_wvl.json"):
    # p = argparse.ArgumentParser()
    # p.add_argument("--config", default=base_config_path, type=str, help="Path to config JSON file")
    # args = p.parse_args()

    cfg = json.load(open(base_config_path))
    base_pop_cfg = cfg["population"]
    var_cfg = cfg.get(varname, {})
    
    # Build one population from base_pop_cfg
    pop = build_population(base_pop_cfg)
    # if varname in ['b_scat', 'b_abs']:
    #     # build_optical_population expects an optics config with a 'type' key
    #     # (morphology). Examples historically used 'morphology' so support that
    #     # by translating into the expected key and filling wavelength/RH grids.
    #     # opt_cfg = dict(var_cfg or {})
    #     # # canonicalize morphology names to registry keys (use underscores)
    #     # morph = opt_cfg.get('morphology', 'core-shell')
    #     # opt_cfg.setdefault('type', str(morph).replace('-', '_'))
    #     # # builder expects 'wvl_grid' and 'rh_grid' keys
    #     # if 'wvl_grid' not in opt_cfg:
    #     #     opt_cfg['wvl_grid'] = var_cfg.get('wvl_grid', [550e-9])
    #     # if 'rh_grid' not in opt_cfg:
    #     #     opt_cfg['rh_grid'] = var_cfg.get('rh_grid', [0.0])
    #     pop = build_optical_population(pop, var_cfg)
    ax = plt.subplot(1,1,1)
    # plot_lines expects an iterable of populations; wrap single pop
    lines,plotdats = plot_lines(ax, (pop,), varname, var_cfg=var_cfg)
    print(plotdats)
    plt.show()
    return lines, ax

varname = 'Nccn'
plot_one_line(varname=varname)


repo_root = Path(__file__).resolve().parent.parent
out = repo_root / "examples" / f"{varname}_lognormal.png"

plt.savefig(out)

    # # Convert example 'population' config schema to a builder-friendly format
    # def config_to_builder_pop(pop_cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    #     # If already appears to be a builder config (has 'type'), return copy
    #     if isinstance(pop_cfg_in, dict) and 'type' in pop_cfg_in:
    #         return dict(pop_cfg_in)
    #     # Do not support the shorthand 'modes' dictionary in this example. Require
    #     # a builder-style configuration (with 'type' and parallel lists such as
    #     # 'N', 'GMD', 'GSD', 'aero_spec_names', 'aero_spec_fracs'). This keeps the
    #     # example unambiguous and avoids implicit unit assumptions.
    #     raise ValueError(
    #         "Unsupported population config format: expected builder-style config with a 'type' key."
    #         " Remove any 'modes' shorthand and provide keys like 'N', 'GMD', 'GSD',"
    #         " 'aero_spec_names', and 'aero_spec_fracs' as parallel lists (see examples/configs/binned_lognormal_bscat_wvl.json)."
    #     )

    # pop_cfg_builder = config_to_builder_pop(pop_cfg)
    # pop_cfg_builder.setdefault("type", "binned_lognormals")
    # pop = build_population(pop_cfg_builder)

    # # Build optics config from variable cfg and population refractive index
    # opt_cfg_builder = dict(var_cfg)
    # opt_cfg_builder.setdefault("type", var_cfg.get("morphology", "homogeneous"))
    # opt_cfg_builder["diameter_units"] = "m"
    # # Force zero dispersion so refractive index stays constant across wavelength
    # opt_cfg_builder.setdefault("alpha_n", 0.0)
    # opt_cfg_builder.setdefault("alpha_k", 0.0)
    # # If population defines a refractive index (e.g., [n, k]) use it
    # refr = pop_cfg.get('refractive_index', pop_cfg.get('refractive_indices', None))
    # if refr is not None:
    #     if isinstance(refr, (list, tuple)) and len(refr) >= 2:
    #         opt_cfg_builder['n_550'] = float(refr[0])
    #         opt_cfg_builder['k_550'] = float(refr[1])
    #     else:
    #         # allow scalar real index
    #         opt_cfg_builder['n_550'] = float(refr)

    # opt_pop = build_optical_population(pop, opt_cfg_builder)

    # # Extract b_scat vs wavelength at RH=0 using OpticalPopulation API
    # wl = np.asarray(opt_cfg_builder.get("wvl_grid", [550e-9]))

    # # Diagnostic: print refractive index used per wavelength to confirm no dispersion
    # print("Refractive indices used by PyParticle:")
    # for i, w in enumerate(wl):
    #     try:
    #         n_val = opt_pop.n_complex[i].real
    #         k_val = opt_pop.n_complex[i].imag
    #         print(f"{w*1e9:.0f} nm: n={n_val:.3f}, k={k_val:.3f}")
    #     except Exception:
    #         # if opt_pop.n_complex not available or shaped differently, skip gracefully
    #         pass
    # rh_val = float(opt_cfg_builder.get("rh_grid", [0.0])[0])
    # # get_optical_coeff returns array across wavelengths when wvl is None
    # b_scat_pp = opt_pop.get_optical_coeff("b_scat", rh=rh_val, wvl=None)
    