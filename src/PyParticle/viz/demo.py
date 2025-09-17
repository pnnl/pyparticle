"""
This file is part of the Flexible Plotting Package.

Role:
- demo: Example script showing how to combine the modules to build a figure
  end-to-end. Demonstrates the workflow: layout -> styling -> plotting ->
  formatting -> save.
"""

import json
from pathlib import Path
import numpy as np
from .layout import make_grid
# use the repo-aware plotting function which understands ParticlePopulation
from .plotting import plot_lines
from .styling import get_colors, get_linestyles
from .formatting import format_axes, add_legend
from PyParticle.population import build_population


def demo_save(path: str = "viz_demo.png"):
  """Create a 3x3 figure showing 3 populations (rows) and 3 variables (cols).

  Rows: three populations (attempt to build PartMC; fallback to altered
  `binned_lognormal` populations if PartMC data aren't available).

  Columns: 'dNdlnD' (size distribution), 'frac_ccn' (fraction CCN),
  'b_ext' (extinction vs wavelength).
  """
  # Load example binned lognormal config to use as a fallback and to
  # generate distinct synthetic populations when PARTMC data are missing.
  cfg_path = Path(__file__).parents[2] / "examples" / "configs" / "binned_lognormal.json"
  try:
    with open(cfg_path, 'r') as fh:
      base_cfg = json.load(fh)
  except Exception:
    base_cfg = None

  # Prepare three population configs. Prefer PARTMC if present; otherwise
  # create three variants of the binned lognormal config.
  pops = []
  pop_names = []

  partmc_attempts = [
    {"type": "partmc", "partmc_dir": str(Path("examples/partmc_run1")), "timestep": 1, "repeat": 0, "n_particles": 400},
    {"type": "partmc", "partmc_dir": str(Path("examples/partmc_run2")), "timestep": 1, "repeat": 0, "n_particles": 400},
    {"type": "partmc", "partmc_dir": str(Path("examples/partmc_run3")), "timestep": 1, "repeat": 0, "n_particles": 400},
  ]

  for ii in range(3):
    # try PARTMC first
    tried_partmc = False
    try:
      cfg = partmc_attempts[ii]
      pop = build_population(cfg)
      pops.append(pop)
      pop_names.append(f"PartMC_{ii+1}")
      tried_partmc = True
    except Exception:
      # fallback to binned lognormal variant
      if base_cfg is None:
        # create a minimal synthetic binned config
        synth = {
          "type": "binned_lognormals",
          "aero_spec_names": [["SO4", "BC"]],
          "aero_spec_fracs": [[0.85, 0.15]],
          "D_min": 0.02,
          "D_max": 1.00,
          "N_bins": 40,
          "N": [1500.0],
          "GMD": [0.10 + 0.05 * ii],
          "GSD": [1.6],
          "D_is_wet": False,
        }
        pop = build_population(synth)
        pops.append(pop)
        pop_names.append(f"synthetic_{ii+1}")
      else:
        # vary the GMD to create different populations
        cfg_mod = dict(base_cfg)
        # ensure lists for N/GMD/GSD
        cfg_mod['GMD'] = [base_cfg.get('GMD', [0.1])[0] + 0.02 * ii]
        cfg_mod['N'] = [base_cfg.get('N', [1500.0])[0]]
        cfg_mod['aero_spec_names'] = base_cfg.get('aero_spec_names')
        cfg_mod['aero_spec_fracs'] = base_cfg.get('aero_spec_fracs')
        cfg_mod['N_bins'] = base_cfg.get('N_bins', 40)
        cfg_mod['D_min'] = base_cfg.get('D_min', 0.02)
        cfg_mod['D_max'] = base_cfg.get('D_max', 1.0)
        pop = build_population(cfg_mod)
        pops.append(pop)
        pop_names.append(f"binned_gmd_{cfg_mod['GMD'][0]:.2f}")

  # Now set up the 3x3 figure: rows = populations, cols = variables
  cols = ['dNdlnD', 'frac_ccn', 'b_ext']
  col_titles = ['Size distribution', 'Fraction CCN', 'Extinction (b_ext)']
  fig, axes = make_grid(3, 3, figsize=(12, 9))

  # colors and linestyles
  colors = get_colors(3)
  linestyles = get_linestyles(3)

  for i, pop in enumerate(pops):
    for j, varname in enumerate(cols):
      ax = axes[i, j]
      var_cfg = None
      if varname == 'b_ext':
        # plot extinction vs wavelength with a few wavelengths
        var_cfg = {
          'wvls': np.array([350e-9, 550e-9, 700e-9]),
          'rh_grid': np.array([0.0, 0.5, 0.8]),
          'morphology': 'core-shell',
          'vs_wvl': True,
          'vs_rh': False
        }

      # plotting function expects a tuple of populations; pass single-element tuple
      try:
        line, labs = plot_lines(varname, (pop,), var_cfg=var_cfg, ax=ax,
                    colors=colors[j] if isinstance(colors, list) else colors,
                    linestyles=linestyles[j] if isinstance(linestyles, list) else linestyles,
                    markers=None)
      except Exception as e:
        # If plotting fails, place a text note
        ax.text(0.5, 0.5, f"Error plotting {varname}: {e}", ha='center', va='center')
        format_axes(ax, title=col_titles[j])
        continue

      # labs may be a list like [xlabel, ylabel]
      xlabel = None
      ylabel = None
      if isinstance(labs, (list, tuple)) and len(labs) >= 1:
        xlabel = labs[0]
      if isinstance(labs, (list, tuple)) and len(labs) >= 2:
        ylabel = labs[1]

      title = f"{pop_names[i]}\n{col_titles[j]}"
      format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
      add_legend(ax)

  fig.tight_layout()
  fig.savefig(path)
  return path


if __name__ == "__main__":
    print("Saving demo figure to:", demo_save())
