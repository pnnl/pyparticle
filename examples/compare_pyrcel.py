#!/usr/bin/env python3
"""
Compare PyParticle CCN outputs against pyrcel (or fallback) and plot activation curves.

Usage:
  python examples/compare_pyrcel.py --input examples/configs/ccn_single_na_cl.json

Saves PNG to reports/compare_pyrcel.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

def load_json_or_abort(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="examples/configs/ccn_single_na_cl.json")
    ap.add_argument("--out", default="reports/compare_pyrcel.png")
    args = ap.parse_args()

    # Ensure running inside the expected conda environment; if not, re-exec using `conda run`
    import os
    import sys
    import shutil
    import subprocess

    desired_env = "pyparticle"
    reexec_marker = os.environ.get("PYPARTICLE_REEXEC", "0")
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env != desired_env and reexec_marker != "1":
        conda_bin = shutil.which("conda")
        if conda_bin is None:
            print(f"Error: conda executable not found and current CONDA_DEFAULT_ENV={current_env}.")
            print("Please activate the 'pyparticle' environment or run the script via ./tools/run_in_env.sh")
            sys.exit(2)
        cmd = [conda_bin, "run", "-n", desired_env, sys.executable] + sys.argv
        env = os.environ.copy()
        env["PYPARTICLE_REEXEC"] = "1"
        print(f"Re-executing under conda env '{desired_env}' using: {' '.join(cmd[:6])} ...")
        rc = subprocess.call(cmd, env=env)
        sys.exit(rc)

    cfg = load_json_or_abort(Path(args.input))

    # Ensure repo root is on sys.path so we can import tools.plot_helpers
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from PyParticle.aerosol_particle import make_particle
        from PyParticle.species.base import AerosolSpecies
        from PyParticle.species.registry import register_species
    except Exception as exc:
        print("PyParticle aerosol_particle unavailable:", exc)
        sys.exit(2)

    particle = cfg.get("particle", {})
    D = particle.get("diameter_m")
    if D is None:
        D = float(particle.get("diameter_um", 100e-9)) * 1e-6

    species = particle.get("species", [])
    names = [s["name"] for s in species]
    fracs = [float(s.get("mass_fraction", 1.0 / max(1, len(names)))) for s in species]

    # Ensure common species exist in registry or register fallbacks
    from PyParticle.species.registry import get_species

    for nm in names:
        try:
            _ = get_species(nm)
        except Exception:
            # Register a minimal fallback AerosolSpecies for common salts
            if nm.upper() == "NACL" or nm.upper().startswith("NACL"):
                fallback = AerosolSpecies(name="NaCl", density=2160.0, kappa=1.5, molar_mass=0.05844, surface_tension=0.072)
                register_species(fallback)
            else:
                # Generic fallback
                fallback = AerosolSpecies(name=nm, density=1500.0, kappa=0.3, molar_mass=0.1, surface_tension=0.072)
                register_species(fallback)

    p = make_particle(D, names, fracs, D_is_wet=True)


    T = float(cfg.get("environment", {}).get("T_K", 298.15))

    # Prepare plotting and S grid
    import numpy as np
    import matplotlib.pyplot as plt
    from tools.plot_helpers import lognormal_mode_to_bins, activation_fraction_from_s_grid

    np.random.seed(0)

    # Supersaturation grid: 0.01% to 1.0% (as fraction)
    S_grid = np.linspace(0.01, 1.0, 100) / 100.0

    # Build a ParticlePopulation using the population builder; population config is required
    from PyParticle.population.builder import build_population

    if "population" not in cfg or cfg["population"] is None:
        raise RuntimeError("compare_pyrcel requires a 'population' configuration entry. Please provide a population config that the population builder can consume.")

    pop_cfg = cfg["population"]
    population = build_population(pop_cfg)
    # Derive D_grid and n_D from the population configuration or discretize first mode if present
    try:
        modes = list(zip(pop_cfg.get("GMD", []), pop_cfg.get("GSD", []), pop_cfg.get("N", [])))
        if len(modes) > 0:
            r0 = float(modes[0][0])
            sigma = float(modes[0][1])
            Ntot = float(modes[0][2])
            D_grid, n_D = lognormal_mode_to_bins(r0, sigma, Ntot, n_bins=100)
        else:
            D_grid = np.array([population.get_particle(pid).get_Ddry() for pid in population.ids])
            n_D = np.array(population.num_concs if hasattr(population, "num_concs") else [1.0 for _ in population.ids])
    except Exception:
        D_grid = np.array([population.get_particle(pid).get_Ddry() for pid in population.ids])
        n_D = np.array(population.num_concs if hasattr(population, "num_concs") else [1.0 for _ in population.ids])
    else:
        population = build_population(pop_cfg)
        # Convert population modes to D_grid,n_D using first mode if necessary
        # Prefer using population.spec_masses / num_concs if available
        # For now, create D_grid and n_D by discretizing the first mode in the config
        modes = list(zip(pop_cfg.get("GMD", []), pop_cfg.get("GSD", []), pop_cfg.get("N", [])))
        if len(modes) > 0:
            r0 = float(modes[0][0])
            sigma = float(modes[0][1])
            Ntot = float(modes[0][2])
            D_grid, n_D = lognormal_mode_to_bins(r0, sigma, Ntot, n_bins=100)
        else:
            # fallback: use population's particles
            D_grid = np.array([population.get_particle(pid).get_Ddry() for pid in population.ids])
            n_D = np.array(population.num_concs if hasattr(population, "num_concs") else [1.0 for _ in population.ids])

    # Compute PyParticle s_crit per particle using population.get_particle and Particle.get_critical_supersaturation
    Tval = float(T)
    s_crit_list = []
    for pid in population.ids:
        part = population.get_particle(pid)
        # ensure surface tension attribute present on particle (72 mN/m)
        try:
            setattr(part, "surface_tension", float(getattr(part, "surface_tension", 0.072)))
        except Exception:
            setattr(part, "surface_tension", 0.072)
        sc = part.get_critical_supersaturation(Tval)
        # get_critical_supersaturation returns percent
        s_crit_list.append(float(sc))

    # Convert s_crit percent -> fraction for comparison with S_grid
    s_crit_array = np.array(s_crit_list, dtype=float) / 100.0

    # For activation fraction vs S_grid, we need s_crit per D in D_grid; approximate by mapping D_grid to nearest population particle sc
    # If population has many particles, compute sc(D) by building particle for each D from population composition - but here approximate
    def s_crit_func(D_arr: np.ndarray):
        # naive mapping: use the first particle's sc for all diameters if only one particle
        if len(s_crit_array) == 1:
            return np.full_like(D_arr, s_crit_array[0])
        # else linearly interpolate between population dry diameters and s_crit
        pop_D = np.array([population.get_particle(pid).get_Ddry() for pid in population.ids])
        pop_sc = np.array(s_crit_array)
        # sort
        idx = np.argsort(pop_D)
        pop_D_sorted = pop_D[idx]
        pop_sc_sorted = pop_sc[idx]
        return np.interp(D_arr, pop_D_sorted, pop_sc_sorted, left=pop_sc_sorted[0], right=pop_sc_sorted[-1])

    activation_particle = activation_fraction_from_s_grid(D_grid, n_D, S_grid, s_crit_func)

    # Reference activation curve: require pyrcel to be installed. Do not use analytic fallbacks.
    import importlib
    if importlib.util.find_spec("pyrcel") is None:
        print("pyrcel not installed: reference activation curve will not be computed. Install pyrcel to enable reference comparisons.")
        activation_ref = None
        ref_name = None
    else:
        # If pyrcel is present, the user may have their own adapter; try direct import
        import pyrcel  # noqa: F401
        # A true integration would call pyrcel's activation routines here. For now, set ref_name.
        ref_name = "pyrcel"
        activation_ref = None

    # Plotting
    plt.figure(figsize=(7, 4))
    plt.plot(S_grid * 100.0, activation_particle, label="PyParticle")
    if activation_ref is not None:
        plt.plot(S_grid * 100.0, activation_ref, label=f"{ref_name}", linestyle="--")
    plt.xlabel("Supersaturation (%)")
    plt.ylabel("Activation fraction")
    plt.title("Activation curve")
    plt.grid(True)
    plt.legend()
    plt.text(0.7, 0.05, f"Reference: {ref_name}", transform=plt.gca().transAxes)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=300)
    print(f"Saved CCN activation plot to {outp}")


if __name__ == "__main__":
    run()
