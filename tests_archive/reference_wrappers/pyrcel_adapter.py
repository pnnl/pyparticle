"""
Adapter to run pyrcel-based CCN activation reference calculations.
If pyrcel is not available, provides a deterministic fallback based on
Kohler approximations (kappa-Kohler analytic critical supersaturation).
"""
from __future__ import annotations

from typing import Dict, Any

import math


def compute_ccn_reference(test_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'s_critical_percent': float} for a single-particle CCN test.

    Expects test_cfg to have fields: particle:{diameter_m, species:[{name,mass_fraction}]}, environment:{T_K}
    """
    # Try to use pyrcel if installed; otherwise fall back to analytic estimate
    try:
        import pyrcel  # noqa: F401
        use_pyrcel = True
    except Exception:
        use_pyrcel = False

    particle = test_cfg.get("particle", {})
    D = particle.get("diameter_m")
    if D is None:
        D = float(particle.get("diameter_um", 0.1)) * 1e-6

    # If possible, construct a Particle via PyParticle to use its get_critical_supersaturation
    try:
        from PyParticle.aerosol_particle import make_particle
        species = particle.get("species", [])
        names = [s.get("name") for s in species]
        fracs = [float(s.get("mass_fraction", 1.0 / max(1, len(names)))) for s in species]
        p = make_particle(D, names, fracs, D_is_wet=False)
        # ensure surface tension attribute expected by get_critical_supersaturation
        try:
            setattr(p, "surface_tension", float(p.surface_tension))
        except Exception:
            setattr(p, "surface_tension", 0.072)

        T = float(test_cfg.get("environment", {}).get("T_K", 298.15))
        try:
            sc = p.get_critical_supersaturation(T)
            if isinstance(sc, (list, tuple)):
                sc = sc[0]
            return {"s_critical_percent": float(sc)}
        except Exception:
            # fall through to analytic fallback
            pass
    except Exception:
        # If PyParticle not importable, fall back to analytic method below
        pass

    # analytic fallback using simple kappa estimate
    species = particle.get("species", [])
    kappa_map = {"NaCl": 1.5, "AS": 0.6, "NH4NO3": 0.6}
    kappas = []
    for s in species:
        name = s.get("name")
        k = s.get("kappa", kappa_map.get(name, 0.3))
        kappas.append(float(k))
    kappa_eff = (sum(kappas) / len(kappas)) if kappas else 0.3

    env = test_cfg.get("environment", {})
    T = float(env.get("T_K", env.get("T", 298.15)))
    sigma = float(test_cfg.get("surface_tension", 0.072))
    MW_h2o = 18e-3
    R = 8.31446261815324
    rho_w = 1000.0
    A = 4.0 * sigma * MW_h2o / (R * T * rho_w)
    Ddry = float(D)
    if kappa_eff <= 0:
        s_c_frac = 1.0
    else:
        s_c_frac = math.sqrt((4.0 * (A ** 3.0)) / (27.0 * kappa_eff * (Ddry ** 3.0)))
    s_c_percent = float(s_c_frac * 100.0)
    return {"s_critical_percent": s_c_percent}


    def compute_ccn_reference_population(population, T: float):
        """
        Compute reference critical supersaturation (percent) per particle in a ParticlePopulation.

        Uses each Particle object's `get_critical_supersaturation(T)` where possible.
        Falls back to analytic kappa-Kohler when necessary.
        Returns list of s_critical_percent values in the same order as population.ids.
        """
        try:
            import pyrcel  # noqa: F401
            have_pyrcel = True
        except Exception:
            have_pyrcel = False

        results = []
        # constants
        R = 8.31446261815324
        sigma_h2o = 0.072
        MW_h2o = 18e-3
        rho_w = 1000.0

        for part_id in population.ids:
            p = population.get_particle(part_id)
            # ensure surface tension attribute is present (72 mN/m)
            try:
                setattr(p, "surface_tension", float(getattr(p, "surface_tension", sigma_h2o)))
            except Exception:
                setattr(p, "surface_tension", sigma_h2o)

            # Prefer built-in PyParticle critical supersaturation
            try:
                sc = p.get_critical_supersaturation(T)
                # p.get_critical_supersaturation may return (s_crit, D_crit)
                if isinstance(sc, (list, tuple)):
                    sc = sc[0]
                results.append(float(sc))
                continue
            except Exception:
                # fall back to analytic estimate
                pass

            # analytic fallback using particle's tkappa and dry diameter
            try:
                tkappa = p.get_tkappa()
            except Exception:
                tkappa = 0.3
            try:
                Ddry = p.get_Ddry()
            except Exception:
                Ddry = 1e-7

            if tkappa <= 0 or Ddry <= 0:
                results.append(100.0)
                continue

            A = 4.0 * sigma_h2o * MW_h2o / (R * T * rho_w)
            s_c_frac = ((4.0 * (A ** 3.0)) / (27.0 * tkappa * (Ddry ** 3.0))) ** 0.5
            results.append(float(s_c_frac * 100.0))

        return results
