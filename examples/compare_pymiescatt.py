#!/usr/bin/env python3
"""
Compare PyParticle optics against PyMieScatt (or fallback) and plot results.

Usage:
  python examples/compare_pymiescatt.py --input examples/configs/optics_homogeneous.json

Saves a PNG to reports/compare_pymiescatt.png
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
    # Ensure running inside the expected conda environment; if not, re-exec using `conda run`
    import os
    import sys
    import shutil
    import subprocess

    desired_env = "pyparticle"
    # marker to avoid infinite re-exec
    reexec_marker = os.environ.get("PYPARTICLE_REEXEC", "0")
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env != desired_env and reexec_marker != "1":
        conda_bin = shutil.which("conda")
        if conda_bin is None:
            print(f"Error: conda executable not found and current CONDA_DEFAULT_ENV={current_env}.")
            print("Please activate the 'pyparticle' environment or run the script via ./tools/run_in_env.sh")
            sys.exit(2)
        # Re-exec under the conda environment
        cmd = [conda_bin, "run", "-n", desired_env, sys.executable] + sys.argv
        env = os.environ.copy()
        env["PYPARTICLE_REEXEC"] = "1"
        print(f"Re-executing under conda env '{desired_env}' using: {' '.join(cmd[:6])} ...")
        rc = subprocess.call(cmd, env=env)
        sys.exit(rc)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="examples/configs/optics_homogeneous.json")
    ap.add_argument("--rh", type=float, default=None, help="RH (fraction) to plot; default first grid value")
    ap.add_argument("--out", default="reports/compare_pymiescatt.png")
    args = ap.parse_args()

    cfg = load_json_or_abort(Path(args.input))

    # Build PyParticle optics
    try:
        from PyParticle.population import build_population
        from PyParticle.optics.builder import build_optical_population
    except Exception as exc:
        print("PyParticle optics builder unavailable:", exc)
        sys.exit(2)

    pop_cfg = cfg.get("population")
    if pop_cfg is None:
        # try standard example population config
        pop_path = Path(__file__).parent / "configs" / "binned_lognormal.json"
        if pop_path.exists():
            import json as _json

            pop_cfg = _json.loads(pop_path.read_text())
        else:
            pop_cfg = {}
    optics_cfg = cfg.get("optics", cfg)

    pop = build_population(pop_cfg)

    # prepare optics config (convert wvl grid)
    optics_cfg = dict(optics_cfg)
    if "wvl_grid_um" in optics_cfg:
        import numpy as np

        optics_cfg["wvl_grid"] = (np.array(optics_cfg.pop("wvl_grid_um")) * 1e-6).tolist()

    opt_pop = build_optical_population(pop, optics_cfg)

    rh_grid = optics_cfg.get("rh_grid", [0.0])
    wvl_grid = optics_cfg.get("wvl_grid", [550e-9])

    rh_sel = args.rh if args.rh is not None else rh_grid[0]

    # PyParticle results
    try:
        bext_pp = opt_pop.get_optical_coeff("ext", rh=rh_sel)
    except Exception as exc:
        print("Error computing PyParticle optics:", exc)
        sys.exit(3)
    # Reference (PyMieScatt) or fallback: compute population-averaged spectra using fixed bins
    import numpy as np
    import matplotlib.pyplot as plt
    from tools.plot_helpers import lognormal_mode_to_bins

    np.random.seed(0)

    # Build a D_grid and n_D from population config (use first mode of binned_lognormals)
    if isinstance(pop_cfg, dict) and "GMD" in pop_cfg:
        try:
            GMD = float(pop_cfg.get("GMD")[0])
            GSD = float(pop_cfg.get("GSD")[0])
            Ntot = float(pop_cfg.get("N")[0])
        except Exception:
            GMD = 100e-9
            GSD = 1.2
            Ntot = 1e8
    else:
        GMD = 100e-9
        GSD = 1.2
        Ntot = 1e8

    D_grid, n_D = lognormal_mode_to_bins(GMD, GSD, Ntot, n_bins=100)

    # wavelengths in meters
    wvl_m = np.array(wvl_grid, dtype=float)
    wvl_um = wvl_m * 1e6

    # Try to use PyMieScatt if available
    import importlib
    use_pymie = importlib.util.find_spec("PyMieScatt") is not None

    b_ext_ref = np.zeros((len(rh_grid), len(wvl_m)))
    b_sca_ref = np.zeros_like(b_ext_ref)
    b_abs_ref = np.zeros_like(b_ext_ref)

    for rr, rh in enumerate(rh_grid):
        # for each diameter compute cross-sections at each wavelength
        Cext_per_D = np.zeros((len(D_grid), len(wvl_m)))
        Csca_per_D = np.zeros_like(Cext_per_D)
        Cabs_per_D = np.zeros_like(Cext_per_D)
        for ii, Dval in enumerate(D_grid):
            area = np.pi * (0.5 * Dval) ** 2
            for ww, lam_m in enumerate(wvl_m):
                if use_pymie:
                    try:
                        from PyMieScatt import MieQ
                        Dnm = float(Dval * 1e9)
                        lamnm = float(lam_m * 1e9)
                        # approximate refractive index from optics_cfg if provided
                        n = float(optics_cfg.get("n_550", 1.54))
                        k = float(optics_cfg.get("k_550", 0.0))
                        m = complex(n, k)
                        out = MieQ(m, lamnm, Dnm, asDict=True, asCrossSection=False)
                        Qext = out.get("Qext", 2.0)
                        Qsca = out.get("Qsca", Qext * 0.9)
                        Qabs = out.get("Qabs", Qext - Qsca)
                        Cext_per_D[ii, ww] = Qext * area
                        Csca_per_D[ii, ww] = Qsca * area
                        Cabs_per_D[ii, ww] = Qabs * area
                        continue
                    except Exception:
                        pass

                # fallback toy-Mie
                x = 2.0 * np.pi * (0.5 * Dval) / lam_m
                Qext = 2.0 * (1.0 - np.exp(-x / 2.0))
                Cext_per_D[ii, ww] = Qext * area
                omega0 = float(optics_cfg.get("single_scatter_albedo", 0.9))
                Csca_per_D[ii, ww] = omega0 * Cext_per_D[ii, ww]
                Cabs_per_D[ii, ww] = (1.0 - omega0) * Cext_per_D[ii, ww]

        # integrate over distribution
        for ww in range(len(wvl_m)):
            b_ext_ref[rr, ww] = np.sum(Cext_per_D[:, ww] * n_D)
            b_sca_ref[rr, ww] = np.sum(Csca_per_D[:, ww] * n_D)
            b_abs_ref[rr, ww] = np.sum(Cabs_per_D[:, ww] * n_D)

    # Plotting multiple RH curves overlaying PyParticle results
    plt.figure(figsize=(10, 6))
    for rr, rh in enumerate(rh_grid):
        try:
            bext_pp_rh = opt_pop.get_optical_coeff("ext", rh=rh)
            bext_pp_rh = np.array(bext_pp_rh)
        except Exception:
            bext_pp_rh = np.array(bext_pp)

        plt.plot(wvl_um, bext_pp_rh, label=f"PyParticle RH={rh}")
        plt.plot(wvl_um, b_ext_ref[rr, :], linestyle="--", label=f"Ref RH={rh}")

    plt.xlabel("Wavelength (um)")
    plt.ylabel("b_ext (m^-1)")
    plt.title("Population-averaged b_ext vs wavelength")
    plt.grid(True)
    plt.legend()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=300)
    print(f"Saved comparison plot to {outp}")


if __name__ == "__main__":
    run()
