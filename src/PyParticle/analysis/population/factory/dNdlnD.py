from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

# todo: move this to a separate "distribution" variable module
# - allow different binning methods (hist, kde, etc)
# - allow different variables (Dwet, tkappa, Cabs, etc.), defined arbitrarily with ParticleVariable
# - allow different weights (dN/dlnD, dmass_bc/dlnD, dCabs/dlnD, etc.)
# - allow 1d, 2d, etc. distributions

@register_variable("dNdlnD")
class DNdlnDVar(PopulationVariable):
    meta = VariableMeta(
        name="dNdlnD",
        axis_names=("D",),
        description="Size distribution dN/dlnD",
        units="m$^{-3}$",
        scale='linear', # dN/dlnD is typically shown on linear scale; diameter itself on log scale
        long_label='Number size distribution',
        short_label='$dN/d\ln D$',
        default_cfg={
            "wetsize": True,
            "normalize": False,
            "method": "hist",
            "N_bins": 80,
            "D_min": 1e-9,
            "D_max": 2e-6,
            "diam_scale": "log",
        },
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        import scipy.stats  # noqa
        edges = np.logspace(np.log10(cfg["D_min"]), np.log10(cfg["D_max"]), cfg["N_bins"] + 1)
        particles = [population.get_particle(pid) for pid in population.ids]
        if cfg["wetsize"]:
            Ds = [p.get_Dwet() for p in particles]
        else:
            Ds = [p.get_Ddry() for p in particles]
        weights = np.asarray(population.num_concs, dtype=float)
        if cfg["normalize"] and weights.sum() > 0:
            weights = weights / weights.sum()
        hist, _ = np.histogram(Ds, bins=edges, weights=weights)
        dln = np.log(edges[1:]) - np.log(edges[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            dNdlnD = np.where(dln > 0, hist / dln, 0.0)
        centers = np.sqrt(edges[:-1] * edges[1:])
        if as_dict:
            return {"D": centers, "dNdlnD": dNdlnD}
        return dNdlnD


def build(cfg=None):
    """Module-level builder for population-style discovery.

    Returns an instantiated variable ready to compute.
    """
    cfg = cfg or {}
    return DNdlnDVar(cfg)
