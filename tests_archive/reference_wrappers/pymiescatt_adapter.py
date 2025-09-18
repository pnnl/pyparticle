"""
Adapter to run PyMieScatt-based optical reference calculations.
If PyMieScatt is not available, provides a deterministic fallback toy-Mie model
that mirrors the fallback behavior implemented in PyParticle optics factories.
"""
from __future__ import annotations

from typing import Dict, Any

import math
import numpy as np


def compute_optics_reference(test_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a simple reference dict with 'b_ext' grid (list of lists) for optics tests.

    Expects test_cfg to contain optics: {rh_grid, wvl_grid_um} and population/preamble.
    """
    try:
        from PyMieScatt import MieQ
        have_pymie = True
    except Exception:
        have_pymie = False

    optics = test_cfg.get("optics", {})
    pop_cfg = test_cfg.get("population") or test_cfg.get("pop_cfg") or test_cfg

    rh_grid = optics.get("rh_grid", [0.0])
    if "wvl_grid_um" in optics:
        wvl_um = np.array(optics.get("wvl_grid_um", [0.55]))
        wvl_m = wvl_um * 1e-6
    else:
        wvl_m = np.array(optics.get("wvl_grid", [550e-9]))

    # For speed/determinism, handle single particle from pop_cfg.ids[0] if possible
    # Fallback: read first species and diameter from population config
    first_diam_m = None
    if isinstance(pop_cfg, dict):
        # try to find a representative diameter
        if "modes" in pop_cfg:
            mode0 = pop_cfg["modes"][0]
            r0 = mode0.get("r0") or mode0.get("r_mean")
            if r0 is not None:
                first_diam_m = float(r0) * 2.0
        if first_diam_m is None:
            first_diam_m = float(pop_cfg.get("diameter_m", pop_cfg.get("D_m", 200e-9)))
    else:
        first_diam_m = 200e-9

    N_rh = len(rh_grid)
    N_wvl = len(wvl_m)
    bext = np.zeros((N_rh, N_wvl))

    for rr, rh in enumerate(rh_grid):
        # simple growth: GF using fallback kappa
        kappa = float(optics.get("fallback_kappa", 0.3))
        rhf = float(rh)
        GF = (1.0 + kappa * rhf / max(1e-6, (1.0 - rhf))) ** (1.0 / 3.0)
        Dwet = first_diam_m * GF
        r = 0.5 * Dwet
        area = math.pi * r * r

        for ww, lam in enumerate(wvl_m):
            # prefer using PyMieScatt if available
            if have_pymie:
                try:
                    Dnm = Dwet * 1e9
                    lamnm = lam * 1e9
                    m = complex(optics.get("n_550", 1.54) + 1j * optics.get("k_550", 0.0))
                    out = MieQ(m, float(lamnm), float(Dnm), asDict=True, asCrossSection=False)
                    Qext = out.get("Qext", 2.0)
                    bext[rr, ww] = Qext * area
                    continue
                except Exception:
                    pass

            # fallback toy-Mie: Qext rises toward 2 with size parameter x
            x = 2.0 * math.pi * r / lam
            Qext = 2.0 * (1.0 - math.exp(-x / 2.0))
            bext[rr, ww] = Qext * area

    return {"b_ext": bext.tolist()}
